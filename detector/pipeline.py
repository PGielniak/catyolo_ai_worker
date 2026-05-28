import logging
import time
import threading
import requests
import cv2
logger = logging.getLogger(__name__)
import base64
import numpy as np
from datetime import datetime
from pathlib import Path
from detector.detectors.occlusion_detectionV2 import OcclusionDetector
from detector.detectors.hailo_runner import HailoRunner

import ctypes

def set_thread_name(name: str):
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass
class DetectionPipeline:
    VLM_QUESTION = "Is the cat attacking a plant. Answer with Yes or No only"
    VLM_COOLDOWN_SECONDS = 30

    def __init__(self, capture, api_base: str):
        self._capture = capture
        self._api_base = api_base
        self._annotated = None
        self._annotated_lock = threading.Lock()
        self._depth_viz = None
        self._depth_viz_lock = threading.Lock()
        self._depth_show = True
        self._depth_show_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.last_saved_sample = None
        self.last_saved_depth = None
        self._depth_samples_dir = Path("/mnt/ssd/home/patryk/pycharm/catyolo_ai_worker/samples")
        self._last_vlm_request = None
        self._last_vlm_answer = None
        self._last_vlm_answer_lock = threading.Lock()

        config = self._fetch_config()
        self._scene = config[0]
        reference_image = self._decode_reference(self._scene["image"]["image"])
        red_zones = self._scene["red_zones"]
        
        self._vlm_prompt_template = self._scene.get("vlm_prompt") or "Is the {class} attacking a plant? Answer with Yes or No only"
        
        all_classes = set()
        for rz in red_zones:
            all_classes.update(rz.get("forbidden_classes", []))
        yolo_classes = list(all_classes) if all_classes else ["cat"]
        logger.info(f"YOLO classes from scene config: {yolo_classes}")

        
        self._occlusion_detector = OcclusionDetector(
            capture=capture,
            reference_image=reference_image,
            red_zones=red_zones,
        )

        self._hailo_runner = HailoRunner(
            capture=capture,
            yolo_classes=yolo_classes
        )

    def start(self):
        self._thread.start()
        self._hailo_runner.start()


    def set_depth_show(self, enabled: bool):
        with self._depth_show_lock:
            self._depth_show = enabled

    def get_depth_show(self) -> bool:
        with self._depth_show_lock:
            return self._depth_show

    def get_annotated(self):
        with self._annotated_lock:
            return self._annotated.copy() if self._annotated is not None else None

    def get_depth_viz(self):
        with self._depth_viz_lock:
            return self._depth_viz.copy() if self._depth_viz is not None else None

    def get_last_vlm_answer(self):
        with self._last_vlm_answer_lock:
            return self._last_vlm_answer

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

        

    def _save_overlap_sample(self, raw_frame, annotated_frame, depth_map, timestamp):
        self._depth_samples_dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp.strftime("%Y%m%d_%H%M%S")

        raw_target = self._depth_samples_dir / f"{ts}_raw.jpg"
        annotated_target = self._depth_samples_dir / f"{ts}_annotated.jpg"
        depth_target = self._depth_samples_dir / f"{ts}_depth.jpg"

        ok1 = cv2.imwrite(str(raw_target), raw_frame)
        ok2 = cv2.imwrite(str(annotated_target), annotated_frame)
        if not ok1 or not ok2:
            logger.error(f"imwrite failed: raw={ok1}, annotated={ok2}")

        if depth_map.dtype != np.uint8:
            min_val = np.min(depth_map)
            max_val = np.max(depth_map)
            if max_val > min_val:
                depth_8u = ((depth_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                depth_8u = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_8u = depth_map
        ok3 = cv2.imwrite(str(depth_target), depth_8u)
        if not ok3:
            logger.error(f"imwrite failed for depth")
        logger.info(f"Overlap samples saved: {ts}")

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


    def _save_vlm_sample(self, annotated_frame, answer, timestamp):
        self._depth_samples_dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp.strftime("%Y%m%d_%H%M%S")
        img_target = self._depth_samples_dir / f"{ts}_vlm_annotated.jpg"
        txt_target = self._depth_samples_dir / f"{ts}_vlm_result.txt"
        ok = cv2.imwrite(str(img_target), annotated_frame)
        if not ok:
            logger.error(f"imwrite failed for VLM sample")
        try:
            with open(txt_target, "w") as f:
                f.write(f"Prompt: {self.VLM_QUESTION}\n")
                f.write(f"Answer: {answer}\n")
                f.write(f"Timestamp: {timestamp.isoformat()}\n")
        except Exception as e:
            logger.error(f"Failed to write VLM result file: {e}")
        logger.info(f"VLM alert sample saved: {ts} — answer: {answer}")

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)

    @staticmethod
    def _detection_overlaps_zones(detection, red_zones, iou_threshold=0.0):
        det_box = (detection.x1, detection.y1, detection.x2, detection.y2)
        for rz in red_zones:
            rz_box = (rz["x"], rz["y"], rz["x"] + rz["width"], rz["y"] + rz["height"])
            if DetectionPipeline._iou(det_box, rz_box) > iou_threshold:
                return True
        return False

    def _check_overlap(self, yolo_result, red_zones):
        """Returns (overlaps: bool, detected_class: str | None)"""
        if yolo_result is None or not yolo_result.detections:
            return False, None
        for det in yolo_result.detections:
            det_box = (det.x1, det.y1, det.x2, det.y2)
            for rz in red_zones:
                rz_box = (rz["x"], rz["y"], rz["x"] + rz["width"], rz["y"] + rz["height"])
                if self._iou(det_box, rz_box) > 0:
                    forbidden = rz.get("forbidden_classes", [])
                    if det.label in forbidden:
                        return True, det.label
        return False, None

    def _run(self):
        set_thread_name("pipeline")
        logger.debug(f"Entered _run method in DetectionPipeline")
        logger.debug(f"{self._scene['scene_id']}")
        logger.debug(f"{self._scene['scene_name']}")
        logger.debug(f"{self._scene['camera_ip_address']}")
        logger.debug(f"{self._scene['camera_port']}")
        logger.debug(f"{self._scene['action_ids']}")
        logger.debug(f"{self._scene['red_zones']}")
        TARGET_FPS = 60
        target_dt = 1.0 / TARGET_FPS
        next_tick = time.monotonic()

        fps_times = []
        red_zones = self._scene["red_zones"]

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

                if yolo_detection.vlm_answer is not None:
                    now_vlm = datetime.now()
                    answer = yolo_detection.vlm_answer
                    with self._last_vlm_answer_lock:
                        self._last_vlm_answer = (answer, now_vlm)
                    logger.info(f"VLM result received: '{answer}'")
                    if "yes" in answer.lower():
                        cv2.putText(annotated, f"VLM: {answer}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self._save_vlm_sample(annotated, answer, now_vlm)

                if yolo_detection.yolo_result is not None:
                    depth_on = self.get_depth_show()
                    self._hailo_runner.set_depth_enabled(depth_on)

                    overlap, detected_class = self._check_overlap(yolo_detection.yolo_result, red_zones)
                    if overlap and detected_class:
                        now_vlm = time.monotonic()
                        if self._last_vlm_request is None or (now_vlm - self._last_vlm_request) >= self.VLM_COOLDOWN_SECONDS:
                            prompt = self._vlm_prompt_template.replace("{class}", detected_class)
                            logger.info(f"Class-aware overlap ({detected_class}) — VLM prompt: {prompt}")
                            self._hailo_runner.request_vlm(frame, prompt)
                            self._last_vlm_request = now_vlm

                    if yolo_detection.depth_map is not None:
                        now_d = datetime.now()
                        if self.last_saved_depth is None or (now_d - self.last_saved_depth).total_seconds() >= 30:
                            self._save_overlap_sample(frame, annotated, yolo_detection.depth_map, now_d)
                            self.last_saved_depth = now_d

                    depth_map = yolo_detection.depth_map
                    if depth_map is not None:
                        min_v, max_v = np.min(depth_map), np.max(depth_map)
                        if max_v > min_v:
                            depth_8u = ((depth_map - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                        else:
                            depth_8u = np.zeros_like(depth_map, dtype=np.uint8)
                        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)
                        with self._depth_viz_lock:
                            self._depth_viz = depth_color



            now_t = time.monotonic()
            fps_times.append(now_t)
            fps_times[:] = [t for t in fps_times if now_t - t < 1.0]
            fps = len(fps_times)
            cv2.putText(annotated, f"FPS: {fps}", (annotated.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            vlm_display = self.get_last_vlm_answer()
            if vlm_display is not None:
                answer, ts = vlm_display
                cv2.putText(annotated, f"VLM: {answer}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            with self._annotated_lock:
                self._annotated = annotated

            next_tick += target_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

        
    def _draw_occlusion(self, annotated, result):
        for rz in result.zones:
            x_start = rz['x']
            y_start = rz['y']
            x_end = x_start + rz['width']
            y_end = y_start + rz['height']

            if rz['occluded']:
                colour = (0, 0, 255)
                status = 'occluded'
            else:
                colour = (0, 165, 255)
                status = 'free'

            cv2.putText(annotated, f"{status} - {rz['occlusion_score']:.2f}",
                        (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), colour, 2)

    def _draw_yolo_detection(self, annotated, detection):
        for det in detection.yolo_result.detections:
            x1 = det.x1
            x2 = det.x2
            y1 = det.y1
            y2 = det.y2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det.label} {det.confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
