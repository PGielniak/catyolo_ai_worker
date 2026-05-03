# detector/detectors/occlusion.py
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from detector.detectors.base import BaseDetector

logger = logging.getLogger(__name__)



@dataclass
class OcclusionResult:
    """Snapshot of occlusion detection state for a single frame."""
    timestamp: float
    zones: list = field(default_factory=list)
    shift: tuple = (0.0, 0.0)
    alignment_confidence: float = 0.0


class OcclusionDetector(BaseDetector):
    """
    Runs occlusion detection in a background thread.
    
    On each frame:
      1. Converts to grayscale + CLAHE
      2. Aligns to the reference frame via phase correlation
      3. Computes SSIM per red zone, flags zones below threshold as occluded
    
    Latest result is exposed via get_latest(); safe to call from other threads.
    """
    ALIGNMENT_INTERVAL = 10.0 
    CLIP_LIMIT = 5
    TILE_GRID_SIZE = (4, 4)
    SSIM_THRESHOLD = 0.4
    MAX_SHIFT_PIXELS = 50
    MIN_ALIGNMENT_CONFIDENCE = 0.1
    IDLE_SLEEP = 0.05
    TARGET_FPS=0.5
    
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
        self._reference_image = reference_image
        self._red_zones = red_zones

        self._cached_shift = (0.0, 0.0)
        self._cached_response = 1.0
        self._last_alignment_time = 0.0
        
        # Pre-compute reference CLAHE once — saves work on every frame
        self._reference_clahe = self._apply_clahe_gray(reference_image,algorithm="none")
        # self._reference_eq_hist = self._apply_clahe_gray(reference_image, algorithm="eq_hist")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        self._lock = threading.Lock()
        self._latest: Optional[OcclusionResult] = None
    
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    
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
        """Load models, fetch config, etc. Called once before loop."""
        pass
    
    def get_latest(self) -> Optional[OcclusionResult]:
        """Return the most recent result, or None if no frame has been processed yet."""
        with self._lock:
            return self._latest
    
    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    
    def _run(self):
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
            
            # Rate cap: sleep until next scheduled tick
            next_tick += target_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We're behind schedule, reset rather than try to catch up
                next_tick = time.monotonic()
    
    def _publish(self, result: OcclusionResult):
        with self._lock:
            self._latest = result
    
    # ------------------------------------------------------------------ #
    # Detection logic
    # ------------------------------------------------------------------ #
    
    def _process(self, frame: np.ndarray) -> OcclusionResult:
        """Compare current frame against reference, return per-zone occlusion status."""
        compare_gray = self._apply_clahe_gray(frame, algorithm="none")
        
        # Re-estimate alignment periodically; otherwise apply cached shift
        now = time.monotonic()
        if now - self._last_alignment_time >= self.ALIGNMENT_INTERVAL:
            _, shift, response = self._align_to_reference(
                self._reference_clahe, compare_gray
            )
            self._cached_shift = shift
            self._cached_response = response
            self._last_alignment_time = now
        
        # Apply cached shift (skip warp when shift is sub-pixel — saves a full warpAffine)
        dx, dy = self._cached_shift
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            compare_aligned = compare_gray
        else:
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            compare_aligned = cv2.warpAffine(
                compare_gray, M,
                (compare_gray.shape[1], compare_gray.shape[0]),
                borderMode=cv2.BORDER_REPLICATE,
            )
        
        zones = self._compare_clahe_ssim(
            self._reference_clahe, compare_aligned, self._red_zones
        )
        
        return OcclusionResult(
            timestamp=time.time(),
            zones=zones,
            shift=self._cached_shift,
            alignment_confidence=self._cached_response,
        )
    
    def process(self, frame) -> OcclusionResult:
      return self._process(frame)

    @classmethod
    def _apply_clahe_gray(cls, img_bgr: np.ndarray, algorithm:str="none") -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.blur(gray, (4, 4))
        
        if algorithm == "eq_hist":
            equalized = cv2.equalizeHist(gray)
        elif algorithm == "clahe":
            equalized = cv2.createCLAHE(
                clipLimit=cls.CLIP_LIMIT, tileGridSize=cls.TILE_GRID_SIZE
            )
        elif algorithm == "none":
            equalized = gray
            logger.debug("no clahe, only blur and gray applied")
        else:
            raise ValueError("unsupported histogram equalization algorithm")
        return equalized
    
    @classmethod
    def _align_to_reference(cls, reference_gray, current_gray):
        DOWNSAMPLE = 8
        
        if DOWNSAMPLE > 1:
            ref_small = cv2.resize(
                reference_gray, None, fx=1/DOWNSAMPLE, fy=1/DOWNSAMPLE,
                interpolation=cv2.INTER_AREA,
            )
            cur_small = cv2.resize(
                current_gray, None, fx=1/DOWNSAMPLE, fy=1/DOWNSAMPLE,
                interpolation=cv2.INTER_AREA,
            )
        else:
            ref_small, cur_small = reference_gray, current_gray
        
        shift, response = cv2.phaseCorrelate(
            np.float32(ref_small), np.float32(cur_small)
        )
        dx = shift[0] * DOWNSAMPLE
        dy = shift[1] * DOWNSAMPLE
        
        # Bail on bad estimates
        if (abs(dx) > cls.MAX_SHIFT_PIXELS
            or abs(dy) > cls.MAX_SHIFT_PIXELS
            or response < cls.MIN_ALIGNMENT_CONFIDENCE):
            return current_gray, (0.0, 0.0), float(response)
        
        # Skip warp if shift is too small to matter (saves the full-frame warpAffine)
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            return current_gray, (float(dx), float(dy)), float(response)
        
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(
            current_gray, M,
            (current_gray.shape[1], current_gray.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, (float(dx), float(dy)), float(response)
    
    @classmethod
    def _compare_clahe_ssim(cls, img1: np.ndarray, img2: np.ndarray, red_zones: list) -> list:
        """Per-zone SSIM comparison. Returns new list of dicts; does not mutate input."""
        results = []
        for rz in red_zones:
            x_start, y_start = rz['x'], rz['y']
            x_end = x_start + rz['width']
            y_end = y_start + rz['height']
            crop1 = img1[y_start:y_end, x_start:x_end]
            crop2 = img2[y_start:y_end, x_start:x_end]
            MAX_DIM = 256
            h, w = crop1.shape
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                crop1 = cv2.resize(crop1, (new_w, new_h), interpolation=cv2.INTER_AREA)
                crop2 = cv2.resize(crop2, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            score, _ = ssim(crop1, crop2, full=True)
            results.append({
                **rz,
                "occluded": bool(score < cls.SSIM_THRESHOLD),
                "occlusion_score": float(score),
                "mean_brightness": float(np.mean(crop2)),
                "original_mean_brightness": float(np.mean(crop1)),
            })
        return results