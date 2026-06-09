import logging
import time
import threading
import requests
import cv2
logger = logging.getLogger(__name__)
import base64
import numpy as np
from datetime import datetime
from detector.detectors.occlusion_detectionV2 import OcclusionDetector
from detector.detectors.hailo_runner import HailoRunner
from detector.events import DetectionEventEmitter, DetectionEvent

import ctypes

def set_thread_name(name: str):
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass
class DetectionPipeline:
    VLM_COOLDOWN_SECONDS = 3
    VLM_MIN_OVERLAP_SECONDS = 1.0
    DEPTH_MARGIN_DEFAULT = 0.20

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
        # Track the last VLM request time per red zone (keyed by zone index) so each
        # zone is throttled independently and one busy zone can't starve another.
        self._last_event_by_zone = {}
        self._overlap_since = {}
        self._vlm_fired_for_zone = set()
        self._last_vlm_answer = None
        self._last_vlm_answer_zone = None
        self._last_vlm_prompt = None
        self._last_vlm_answer_lock = threading.Lock()
        self._latest_occlusion_result = None
        self._reference_depths_ok = False
        self._reference_depths: dict[int, float] = {}

        config = self._fetch_config()
        self._scene = config[0]
        reference_image = self._decode_reference(self._scene["image"]["image"])
        red_zones = self._scene["red_zones"]

        self._scene_prompt = self._scene.get("scene_prompt")
        self._scene_prompt_interval = self._scene.get("scene_prompt_interval")
        self._scene_prompt_action_ids = self._scene.get("scene_prompt_action_ids")
        self._last_global_vlm_time = 0.0
        self._last_global_vlm_answer = None
        self._last_global_vlm_answer_time = None
        self._pending_vlm_is_global = False
        self._last_processed_vlm_result = None

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
            yolo_classes=yolo_classes,
            reference_image=reference_image,
            red_zones=red_zones,
        )

        self._detection_events = DetectionEventEmitter()

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
        """Returns (zone_index, detected_class) for the first forbidden overlap, else (None, None)."""
        if yolo_result is None or not yolo_result.detections:
            return None, None
        for det in yolo_result.detections:
            det_box = (det.x1, det.y1, det.x2, det.y2)
            for idx, rz in enumerate(red_zones):
                rz_box = (rz["x"], rz["y"], rz["x"] + rz["width"], rz["y"] + rz["height"])
                if self._iou(det_box, rz_box) > 0:
                    forbidden = rz.get("forbidden_classes", [])
                    if det.label in forbidden:
                        return idx, det.label
        return None, None

    def _overlapping_zone_indices(self, yolo_result, red_zones):
        """Returns {zone_index: first_matching_class} for zones with a forbidden-class bbox overlap."""
        if yolo_result is None or not yolo_result.detections:
            return {}
        hits = {}
        for det in yolo_result.detections:
            det_box = (det.x1, det.y1, det.x2, det.y2)
            for idx, rz in enumerate(red_zones):
                if idx in hits:
                    continue
                rz_box = (rz["x"], rz["y"], rz["x"] + rz["width"], rz["y"] + rz["height"])
                if self._iou(det_box, rz_box) > 0:
                    forbidden = rz.get("forbidden_classes", [])
                    if det.label in forbidden:
                        hits[idx] = det.label
        return hits

    def _any_zone_wants_depth(self) -> bool:
        for rz in self._scene.get("red_zones", []):
            if rz.get("depth_enabled"):
                return True
        return False

    def _check_depth_match(self, zone_idx: int, depth_map, detection_bbox) -> bool:
        if not self._reference_depths_ok or zone_idx not in self._reference_depths:
            return True
        ref_depth = self._reference_depths[zone_idx]
        zone = self._scene["red_zones"][zone_idx]
        margin = zone.get("depth_margin") or self.DEPTH_MARGIN_DEFAULT

        x1 = max(0, int(detection_bbox[0]))
        y1 = max(0, int(detection_bbox[1]))
        x2 = min(depth_map.shape[1], int(detection_bbox[2]))
        y2 = min(depth_map.shape[0], int(detection_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return True
        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            return True
        bbox_median = float(np.median(crop))
        diff = abs(bbox_median - ref_depth) / max(abs(ref_depth), 1e-8)
        if diff > margin:
            logger.info(f"Depth gate blocked — zone {zone_idx}: "
                        f"ref={ref_depth:.3f} bbox={bbox_median:.3f} diff={diff:.3f} > margin={margin:.3f}")
            return False
        return True

    def _run(self):
        set_thread_name("pipeline")
        logger.debug(f"Entered _run method in DetectionPipeline")
        logger.debug(f"{self._scene['scene_id']}")
        logger.debug(f"{self._scene['scene_name']}")
        logger.debug(f"{self._scene['camera_ip_address']}")
        logger.debug(f"{self._scene['camera_port']}")
        logger.debug(f"{self._scene['red_zones']}")
        TARGET_FPS = 30
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
                self._latest_occlusion_result = result
                self._draw_occlusion(annotated, result)

            yolo_detection = self._hailo_runner.get_latest()

            if yolo_detection is not None:
                self._draw_yolo_detection(annotated, yolo_detection)

                if yolo_detection.vlm_answer is not None and yolo_detection is not self._last_processed_vlm_result:
                    self._last_processed_vlm_result = yolo_detection
                    now_vlm = datetime.now()
                    answer = yolo_detection.vlm_answer
                    with self._last_vlm_answer_lock:
                        self._last_vlm_answer = (answer, now_vlm)
                    logger.info(f"VLM result received: '{answer}'")
                    self._draw_vlm_answer(annotated, answer, self._last_vlm_answer_zone, question=self._last_vlm_prompt)

                    if self._pending_vlm_is_global:
                        self._pending_vlm_is_global = False
                        if answer == "Yes":
                            self._detection_events.emit(DetectionEvent(
                                annotated_image=annotated.copy(),
                                trigger="global_prompt",
                                vlm_prompt=self._last_vlm_prompt or "",
                                vlm_answer=answer,
                                timestamp=now_vlm,
                                zone=None,
                            ))
                    elif "yes" in answer.lower():
                        det_class = self._last_vlm_answer_zone.get("forbidden_classes", [])[0] if isinstance(self._last_vlm_answer_zone, dict) and self._last_vlm_answer_zone.get("forbidden_classes") else None
                        self._detection_events.emit(DetectionEvent(
                            annotated_image=annotated.copy(),
                            trigger="vlm_yes",
                            detected_class=det_class,
                            vlm_prompt=self._last_vlm_prompt or "",
                            vlm_answer=answer,
                            timestamp=now_vlm,
                            zone=self._last_vlm_answer_zone,
                        ))

                if yolo_detection.yolo_result is not None:
                    depth_on = self.get_depth_show() or self._any_zone_wants_depth()
                    self._hailo_runner.set_depth_enabled(depth_on)

                    if not self._reference_depths_ok:
                        ready, ref_depths = self._hailo_runner.get_reference_depths(timeout=0.0)
                        if ready:
                            self._reference_depths = ref_depths
                            self._reference_depths_ok = True

                    now_vlm = time.monotonic()
                    overlapping_now = self._overlapping_zone_indices(yolo_detection.yolo_result, red_zones)

                    for zi in list(self._overlap_since):
                        if zi not in overlapping_now:
                            self._overlap_since.pop(zi, None)
                            self._vlm_fired_for_zone.discard(zi)

                    for zi, detected_class in overlapping_now.items():
                        if zi not in self._overlap_since:
                            self._overlap_since[zi] = now_vlm
                        elif zi not in self._vlm_fired_for_zone:
                            elapsed = now_vlm - self._overlap_since[zi]
                            if elapsed >= self.VLM_MIN_OVERLAP_SECONDS:
                                last_request = self._last_event_by_zone.get(zi)
                                if last_request is None or (now_vlm - last_request) >= self.VLM_COOLDOWN_SECONDS:
                                    zone = red_zones[zi]

                                    zone_occluded = False
                                    if (self._latest_occlusion_result is not None and
                                        zi < len(self._latest_occlusion_result.zones)):
                                        rz = self._latest_occlusion_result.zones[zi]
                                        if rz.get("occluded"):
                                            zone_occluded = True
                                            logger.info(f"Occlusion gate blocked — zone {zi} is occluded "
                                                        f"(score={rz.get('occlusion_score', 0):.2f})")
                                    if zone_occluded:
                                        continue

                                    if not zone.get("vlm_prompt"):
                                        if zone.get("depth_enabled") and yolo_detection.depth_map is not None:
                                            det_box = None
                                            for det in yolo_detection.yolo_result.detections:
                                                if det.label == detected_class:
                                                    det_box = (det.x1, det.y1, det.x2, det.y2)
                                                    break
                                            if det_box and not self._check_depth_match(zi, yolo_detection.depth_map, det_box):
                                                continue
                                            trigger = "depth_match"
                                        else:
                                            trigger = "overlap"
                                        logger.info(f"DetectionEvent trigger — zone {zi} overlapped for {elapsed:.1f}s, class={detected_class}, trigger={trigger}")
                                        self._detection_events.emit(DetectionEvent(
                                            annotated_image=annotated.copy(),
                                            trigger=trigger,
                                            detected_class=detected_class,
                                            timestamp=datetime.now(),
                                            zone=zone,
                                        ))
                                        self._last_event_by_zone[zi] = now_vlm
                                        self._last_vlm_answer_zone = zone
                                        self._last_vlm_prompt = ""
                                        self._vlm_fired_for_zone.add(zi)
                                        self._pending_vlm_is_global = False
                                        continue

                                    if zone.get("depth_enabled") and yolo_detection.depth_map is not None:
                                        det_box = None
                                        for det in yolo_detection.yolo_result.detections:
                                            if det.label == detected_class:
                                                det_box = (det.x1, det.y1, det.x2, det.y2)
                                                break
                                        if det_box and not self._check_depth_match(zi, yolo_detection.depth_map, det_box):
                                            continue

                                    prompt_template = zone.get("vlm_prompt") or "Is the {class} attacking a plant?"
                                    prompt = prompt_template.replace("{class}", detected_class)
                                    logger.info(f"VLM trigger — zone {zi} overlapped for {elapsed:.1f}s, class={detected_class}")
                                    self._hailo_runner.request_vlm(frame, zone=zone, detected_class=detected_class)
                                    self._last_event_by_zone[zi] = now_vlm
                                    self._last_vlm_answer_zone = zone
                                    self._last_vlm_prompt = prompt
                                    self._vlm_fired_for_zone.add(zi)
                                    self._pending_vlm_is_global = False

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

            if (self._scene_prompt
                    and self._scene_prompt_interval
                    and self._scene_prompt_interval > 0):
                now_mono = time.monotonic()
                if (now_mono - self._last_global_vlm_time) >= self._scene_prompt_interval:
                    logger.info(f"Global VLM trigger — interval={self._scene_prompt_interval}s")
                    self._hailo_runner.request_vlm(
                        frame,
                        zone=None,
                        detected_class=None,
                        is_global=True,
                        global_prompt=self._scene_prompt,
                    )
                    self._last_global_vlm_time = now_mono
                    self._last_vlm_answer_zone = None
                    self._last_vlm_prompt = self._scene_prompt
                    self._pending_vlm_is_global = True



            now_t = time.monotonic()
            fps_times.append(now_t)
            fps_times[:] = [t for t in fps_times if now_t - t < 1.0]
            fps = len(fps_times)
            cv2.putText(annotated, f"FPS: {fps}", (annotated.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            vlm_display = self.get_last_vlm_answer()
            if vlm_display is not None:
                answer, ts = vlm_display
                self._draw_vlm_answer(annotated, answer, self._last_vlm_answer_zone,
                                      question=self._last_vlm_prompt, color=(255, 255, 255))

            with self._annotated_lock:
                self._annotated = annotated

            next_tick += target_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

        
    def _draw_vlm_answer(self, annotated, answer, zone, question=None, color=(255, 255, 255)):
        x = zone.get("x", 10) if zone else 10
        y = zone.get("y", 30) if zone else 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        lines = []
        if question:
            lines.append(("Q: " + question, (200, 200, 200)))
        lines.append(("VLM: " + answer, color))

        text_x = x + 4
        text_y = max(y - 22, 14)
        for text, clr in reversed(lines):
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            text_y -= th + 6
            cv2.rectangle(annotated, (text_x - 3, text_y - 2),
                          (text_x + tw + 3, text_y + th + baseline + 2),
                          (0, 0, 0), -1)
            cv2.putText(annotated, text, (text_x, text_y + th),
                        font, scale, clr, thickness)

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
