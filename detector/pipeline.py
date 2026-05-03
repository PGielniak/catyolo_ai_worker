import logging
import time
import threading
import requests
import cv2
logger = logging.getLogger(__name__)
import base64
import numpy as np
from scripts.occlusion import check_occlusion
from datetime import datetime
from pathlib import Path
from detector.detectors.occlusion_detectionV2 import OcclusionDetector
from detector.detectors.hailo_runner import HailoRunner

import ctypes

def set_thread_name(name: str):
    """Set the OS-visible thread name. Linux only, max 15 chars."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)  # 15 = PR_SET_NAME
    except Exception:
        pass
class DetectionPipeline:
    def __init__(self, capture, api_base: str):
        self._capture = capture
        self._api_base = api_base
        self._annotated = None
        self._annotated_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.last_saved_sample = None 

        config = self._fetch_config()
        self._scene = config[0]  # keep around for logging / red zone access in _run
        reference_image = self._decode_reference(self._scene["image"]["image"])
        red_zones = self._scene["red_zones"]

        
        self._occlusion_detector = OcclusionDetector(
            capture=capture,
            reference_image=reference_image,
            red_zones=red_zones,
        )

        self._hailo_runner = HailoRunner(
            capture=capture,
            yolo_classes=["cat"] # TODO pass from scene config
        )

    def start(self):
        self._thread.start()
        self._hailo_runner.start()


    def get_annotated(self):
        with self._annotated_lock:
            return self._annotated.copy() if self._annotated is not None else None

    def _fetch_config(self) -> dict:
        try:
            logger.info(f"Fetching config from {self._api_base}/scene")
            r = requests.get(f"{self._api_base}/scene")
            r.raise_for_status()
            logger.debug(f"{r.json()}")
            return r.json()
        except Exception as e:
            logger.error(f"Failed to fetch config: {e}")
            raise

    @staticmethod
    def _decode_reference(b64: str) -> np.ndarray:
        try:
            logger.debug(f"starting _decode_reference")
            img_bytes = base64.b64decode(b64)
            logger.debug(f"img_bytes length: {len(img_bytes)}")
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("reference image is None")
            return img
        except Exception as e:
            logger.error(f"Failed to decode reference image")

        

    def _save_sample(self,img_raw, img_annotated, timestamp):
        if img_annotated is None:
            logger.error("img_annotated is None")
            return
        
        output_path = Path("/mnt/ssd/home/patryk/pycharm/catyolo_ai_worker/occl_yolo_samples")
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = output_path / f"{timestamp_str}_annotated.jpg"
        target_raw = output_path / f"{timestamp_str}_raw.jpg"
        
        logger.info(f"saving: type={type(img_annotated).__name__}, "
                    f"shape={getattr(img_annotated, 'shape', 'N/A')}, "
                    f"dtype={getattr(img_annotated, 'dtype', 'N/A')}, "
                    f"contiguous={getattr(img_annotated, 'flags', None) and img_annotated.flags['C_CONTIGUOUS']}")

        logger.info(f"saving: type={type(img_raw).__name__}, "
                    f"shape={getattr(img_raw, 'shape', 'N/A')}, "
                    f"dtype={getattr(img_raw, 'dtype', 'N/A')}, "
                    f"contiguous={getattr(img_raw, 'flags', None) and img_raw.flags['C_CONTIGUOUS']}")
        
        try:
            ok = cv2.imwrite(str(target), img_annotated)
            ok2 = cv2.imwrite(str(target_raw),img_raw)
            if not ok:
                logger.error(f"imwrite returned False for {target}")
            if not ok2:
                logger.error(f"imwrite returned False for {target_raw}")
        except cv2.error as e:
            logger.error(f"imwrite raised error: {e}")

    def _run(self):
        set_thread_name("pipeline")
        logger.debug(f"Entered _run method in DetectionPipeline")
        logger.debug(f"{self._scene['scene_id']}")
        logger.debug(f"{self._scene['scene_name']}")
        logger.debug(f"{self._scene['camera_ip_address']}")
        logger.debug(f"{self._scene['camera_port']}")
        logger.debug(f"{self._scene['action_ids']}")
        logger.debug(f"{self._scene['red_zones']}")
        TARGET_FPS = 10  # 10 FPS is plenty for monitoring stream
        target_dt = 1.0 / TARGET_FPS
        next_tick = time.monotonic()

        last_save_minute = None

        while True:
            frame = self._capture.get()
            if frame is None:
                time.sleep(0.05)
                continue

            annotated = frame.copy()

            
            result = self._occlusion_detector.process(frame)

            if result is not None:
                self._draw_occlusion(annotated, result)

            yolo_detection = self._hailo_runner.get_latest()

            if yolo_detection is not None:
                self._draw_yolo_detection(annotated, yolo_detection)
                if len(yolo_detection.yolo_result.detections) > 0:
                    
                    current_time = datetime.now()
                    if self.last_saved_sample is None or (current_time - self.last_saved_sample).total_seconds() >= 600:  # 600 seconds = 10 minutes
                        logger.info("Cat detected - saving sample!")
                        self._save_sample(frame, annotated, current_time)
                        self.last_saved_sample = current_time

            # Periodic sample save (once per :00 / :30 minute)
            now = datetime.now()
            current_minute = (now.hour, now.minute)
            if now.minute in (0, 30) and current_minute != last_save_minute:
                self._save_sample(frame, annotated, now)
                last_save_minute = current_minute

            with self._annotated_lock:
                self._annotated = annotated

            next_tick += target_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

            # time.sleep(0.033)  # ~30 fps cap on the annotator loop
        
    def _draw_occlusion(self, annotated, result):
        # Global alignment info, top-left
        # shift = result.shift
        # cv2.putText(
        #     annotated,
        #     f"shift: ({shift[0]:.1f}, {shift[1]:.1f}) conf: {result.alignment_confidence:.2f}",
        #     (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        # )

        for rz in result.zones:
            x_start = rz['x']
            y_start = rz['y']
            x_end = x_start + rz['width']
            y_end = y_start + rz['height']

            if rz['occluded']:
                colour = (0, 0, 255)  # red
                status = 'occluded'
            else:
                colour = (0, 165, 255)  # orange
                status = 'free'

            cv2.putText(annotated, f"{status} - {rz['occlusion_score']:.2f}",
                        (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), colour, 2)

    def _draw_yolo_detection(self, annotated, detection):
        """Draw a single YOLO detection on the image."""

        for det in detection.yolo_result.detections:
            x1 = det.x1
            x2 = det.x2
            y1 = det.y1
            y2 = det.y2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det.label} {det.confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


