# detector/detectors/occlusion.py
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from detector.detectors.base import BaseDetector
import ctypes
logger = logging.getLogger(__name__)


@dataclass
class OcclusionResult:
    """Snapshot of zone tracking state for a single frame."""
    timestamp: float
    zones: list = field(default_factory=list)


class OcclusionDetector(BaseDetector):
    """
    Tracks plant zones via template matching, with motion priors to prevent
    drift onto false matches (hands, cats, similar-textured objects).
    
    Per frame:
      1. Search for each zone's reference template in a window around its
         current position.
      2. Update zone coordinates only if the match is:
           - Confident enough (POSITION_UPDATE_THRESHOLD)
           - Not too far from the previous position (MAX_UPDATE_DISTANCE)
           - Not too far from the original position (ABSOLUTE_DRIFT_LIMIT)
      3. Report match score; below OCCLUSION_THRESHOLD = occluded.
    
    The zone box stays anchored to the plant's true location even when
    something is in front of it.
    """
    
    # Detection thresholds
    OCCLUSION_THRESHOLD = 0.6         # match score below this = "occluded"
    POSITION_UPDATE_THRESHOLD = 0.8   # match score required to consider moving the zone
    
    # Motion priors
    MAX_UPDATE_DISTANCE = 5           # pixels — max single-step movement
    ABSOLUTE_DRIFT_LIMIT = 30         # pixels — max total drift from original position
    
    # Search behavior
    ZONE_SEARCH_MARGIN = 30        # pixels around current zone to search
    
    # Loop / preprocessing
    DOWNSAMPLE = 4                  # 1080p → 540p before processing
    TARGET_FPS = 0.2
    IDLE_SLEEP = 0.05
    
    def __init__(self, capture, reference_image: np.ndarray, red_zones: list):
        """
        Args:
            capture: frame source with a .get() method returning the latest BGR frame
            reference_image: BGR image (numpy array) to compare against
            red_zones: list of dicts with keys 'x', 'y', 'width', 'height' (and optional 'id')
        """
        if reference_image is None:
            raise ValueError("reference_image is required")
        if not red_zones:
            logger.warning("OcclusionDetector created with no red_zones; will produce empty results")
        
        self._capture = capture
        self._red_zones_original = red_zones  # for reference / debugging
        
        # Working zones in downsampled coordinate space
        # Each zone tracks both its current position (x,y) and its original anchor (original_x,y)
        self._red_zones = []
        for rz in red_zones:
            scaled_x = rz['x'] // self.DOWNSAMPLE
            scaled_y = rz['y'] // self.DOWNSAMPLE
            self._red_zones.append({
                **rz,
                'x': scaled_x,
                'y': scaled_y,
                'width': rz['width'] // self.DOWNSAMPLE,
                'height': rz['height'] // self.DOWNSAMPLE,
                'original_x': scaled_x,
                'original_y': scaled_y,
            })
        
        # Pre-compute downsampled reference grayscale + per-zone templates
        h, w = reference_image.shape[:2]
        ref_small = cv2.resize(
            reference_image, (w // self.DOWNSAMPLE, h // self.DOWNSAMPLE),
            interpolation=cv2.INTER_AREA,
        )
        ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
        
        self._reference_zone_crops = {}
        for rz in self._red_zones:
            crop = ref_gray[
                rz['y']:rz['y'] + rz['height'],
                rz['x']:rz['x'] + rz['width']
            ]
            self._reference_zone_crops[id(rz)] = crop
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[OcclusionResult] = None
    
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    


    def _set_thread_name(name):
        """Set the OS-visible thread name (Linux only, max 15 chars)."""
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(15, name.encode()[:15], 0, 0, 0)  # 15 = PR_SET_NAME
        except Exception:
            pass
    def start(self):
        if self._thread is not None:
            logger.warning("OcclusionDetector already started")
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="occlusion"
        )
        self._thread.start()
    
    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
    
    def _setup(self):
        """No-op — all setup happens in __init__."""
        pass
    
    def get_latest(self) -> Optional[OcclusionResult]:
        with self._lock:
            return self._latest
    
    def process(self, frame) -> OcclusionResult:
        return self._process(frame)
    
    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    
    def _run(self):
        _set_thread_name("occlusion") 
        logger.info("OcclusionDetector running (zones=%d)", len(self._red_zones))
        target_dt = 1.0 / self.TARGET_FPS
        next_tick = time.monotonic()
        
        while not self._stop_event.is_set():
            frame = self._capture.get()
            if frame is None:
                time.sleep(self.IDLE_SLEEP)
                continue
            
            try:
                result = self._process(frame)
                self._publish(result)
            except Exception:
                logger.exception("OcclusionDetector processing error")
            
            next_tick += target_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    
    def _publish(self, result: OcclusionResult):
        with self._lock:
            self._latest = result
    
    # ------------------------------------------------------------------ #
    # Detection logic
    # ------------------------------------------------------------------ #
    
    def _process(self, frame: np.ndarray) -> OcclusionResult:
        if self.DOWNSAMPLE > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame, (w // self.DOWNSAMPLE, h // self.DOWNSAMPLE),
                interpolation=cv2.INTER_AREA,
            )
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        zones = self._track_zones(current_gray)
        
        return OcclusionResult(timestamp=time.time(), zones=zones)
    
    def _track_zones(self, current_gray: np.ndarray) -> list:
        """
        Find each zone's template in current frame.
        Update coords only if match passes confidence + motion-prior checks.
        Always report current state with match score.
        """
        h, w = current_gray.shape
        margin = self.ZONE_SEARCH_MARGIN
        results = []
        
        for rz in self._red_zones:
            template = self._reference_zone_crops[id(rz)]
            tpl_h, tpl_w = template.shape
            
            # Search window around current zone position
            sx1 = max(0, rz['x'] - margin)
            sy1 = max(0, rz['y'] - margin)
            sx2 = min(w, rz['x'] + tpl_w + margin)
            sy2 = min(h, rz['y'] + tpl_h + margin)
            search_area = current_gray[sy1:sy2, sx1:sx2]
            
            # Bail if search area is too small (zone near frame edge)
            if search_area.shape[0] < tpl_h or search_area.shape[1] < tpl_w:
                results.append(self._zone_result(rz, score=0.0, occluded=True))
                continue
            
            result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, match_score, _, max_loc = cv2.minMaxLoc(result)
            
            # Decide whether to accept the new position
            if match_score >= self.POSITION_UPDATE_THRESHOLD:
                new_x = sx1 + max_loc[0]
                new_y = sy1 + max_loc[1]
                
                step = max(abs(new_x - rz['x']), abs(new_y - rz['y']))
                drift_x = abs(new_x - rz['original_x'])
                drift_y = abs(new_y - rz['original_y'])
                
                if step > self.MAX_UPDATE_DISTANCE:
                    logger.debug(
                        "rejecting large step %dpx for zone @ (%d,%d), score=%.2f",
                        step, rz['x'], rz['y'], match_score
                    )
                elif drift_x > self.ABSOLUTE_DRIFT_LIMIT or drift_y > self.ABSOLUTE_DRIFT_LIMIT:
                    logger.debug(
                        "rejecting drifted match (%d,%d) → (%d,%d), score=%.2f",
                        rz['original_x'], rz['original_y'], new_x, new_y, match_score
                    )
                else:
                    if new_x != rz['x'] or new_y != rz['y']:
                        logger.debug(
                            "zone moved: (%d,%d) -> (%d,%d) score=%.2f",
                            rz['x'], rz['y'], new_x, new_y, match_score
                        )
                    rz['x'] = new_x
                    rz['y'] = new_y
            
            occluded = match_score < self.OCCLUSION_THRESHOLD
            results.append(self._zone_result(rz, score=match_score, occluded=occluded))
        
        return results
    
    def _zone_result(self, rz: dict, score: float, occluded: bool) -> dict:
        """Build the result dict for one zone, with coordinates in original-resolution space."""
        return {
            **rz,
            'x': rz['x'] * self.DOWNSAMPLE,
            'y': rz['y'] * self.DOWNSAMPLE,
            'width': rz['width'] * self.DOWNSAMPLE,
            'height': rz['height'] * self.DOWNSAMPLE,
            "occluded": bool(occluded),
            "occlusion_score": float(score),
        }