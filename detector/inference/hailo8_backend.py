import ctypes
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from detector.capture import FrameCapture
from detector.geometry import polygon_bounding_box, zone_to_polygon
from detector.inference.preprocessing import COCO_CLASSES, bbox_unmap, letterbox
from detector.inference.protocols import (
    BackendCapabilities,
    HailoResult,
    InferenceBackend,
    YoloDetection,
    YoloResult,
)

logger = logging.getLogger(__name__)


def _set_thread_name(name: str) -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass


class Hailo8Backend(InferenceBackend):
    """Hailo-8 backend: YOLO object detection + optional depth. No VLM support."""

    RELOAD_SETTLE_SECONDS: float = 2.0
    SETUP_TIMEOUT: float = 60.0
    IDLE_SLEEP: float = 0.05

    def __init__(
        self,
        capture: FrameCapture,
        yolo_classes: list[str],
        hef_config: dict,
        reference_image: Optional[np.ndarray] = None,
        red_zones: Optional[list] = None,
        shared_device: Any = None,
    ):
        self._capture = capture
        self.yolo_classes = yolo_classes
        self._class_ids = [COCO_CLASSES.index(c) for c in yolo_classes]
        self._reference_image = reference_image
        self._red_zones = red_zones
        self._confidence_threshold = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4"))

        self._yolo_path = Path(hef_config["yolo"]["path"])
        self._depth_path = Path(hef_config["depth"]["path"]) if "depth" in hef_config else None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[HailoResult] = None
        self._setup_complete = threading.Event()

        # Device ownership: shared (multi-camera, owned by main()) vs owned
        # (legacy/test path). See hailo10_backend.py for the full rationale.
        self._device = None
        self._shared_device = shared_device
        self._owns_device = shared_device is None

        self._yolo_infer_model = None
        self._yolo_configured_infer_model = None
        self._desired_h: Optional[int] = None
        self._desired_w: Optional[int] = None

        self._fastdepth_infer_model = None
        self._fastdepth_configured_infer_model = None
        self._fastdepth_desired_h: Optional[int] = None
        self._fastdepth_desired_w: Optional[int] = None
        self._fastdepth_c: Optional[int] = None

        self._depth_enabled = False
        self._depth_lock = threading.Lock()

        self._reference_depths: dict[int, float] = {}
        self._reference_depths_ready = threading.Event()

        has_depth = self._depth_path is not None
        self._capabilities = BackendCapabilities(
            supports_vlm=False,
            supports_depth=has_depth,
            max_concurrent_streams=3,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def start(self) -> None:
        if self._thread is not None:
            logger.warning("Hailo8Backend already started")
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="hailo")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._teardown()

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        return self._setup_complete.wait(timeout=timeout)

    def get_latest(self) -> Optional[HailoResult]:
        with self._lock:
            return self._latest

    def set_depth_enabled(self, enabled: bool) -> None:
        with self._depth_lock:
            self._depth_enabled = enabled

    def get_reference_depths(self, timeout: float = 0.0) -> tuple[bool, dict[int, float]]:
        ready = self._reference_depths_ready.wait(timeout=timeout)
        with self._lock:
            return ready, dict(self._reference_depths)

    # ------------------------------------------------------------------ #
    # Setup / teardown
    # ------------------------------------------------------------------ #

    def _setup(self) -> None:
        if self._shared_device is not None:
            # Multi-camera path: attach to the VDevice owned by main().
            self._device = self._shared_device
            logger.info("Attaching to shared VDevice (owned by main)")
        else:
            # Legacy/test path: own the VDevice lifecycle.
            from hailo_platform import Device, HailoSchedulingAlgorithm, VDevice

            with Device() as dev:
                ids = dev.scan()
                logger.info("Available Hailo devices: %s", ids)
            if not ids:
                raise RuntimeError("No Hailo devices found")

            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.group_id = "SHARED"
            self._device = VDevice(params)

        self._setup_yolo()
        if self._depth_path is not None:
            self._setup_depth()

    def _setup_yolo(self) -> None:
        try:
            self._yolo_infer_model = self._device.create_infer_model(str(self._yolo_path))
            input_name = self._yolo_infer_model.input_names[0]
            self._desired_h, self._desired_w, _ = self._yolo_infer_model.input(input_name).shape
            self._yolo_configured_infer_model = self._yolo_infer_model.configure()
            logger.info("YOLO configured (%dx%d)", self._desired_h, self._desired_w)
        except Exception as e:
            logger.error("Error setting up YOLO: %s", e)
            raise

    def _setup_depth(self) -> None:
        try:
            self._fastdepth_infer_model = self._device.create_infer_model(str(self._depth_path))
            input_name = self._fastdepth_infer_model.input_names[0]
            self._fastdepth_desired_h, self._fastdepth_desired_w, self._fastdepth_c = (
                self._fastdepth_infer_model.input(input_name).shape
            )
            self._fastdepth_configured_infer_model = self._fastdepth_infer_model.configure()
            logger.info("Depth configured (%dx%d)", self._fastdepth_desired_h, self._fastdepth_desired_w)
        except Exception as e:
            logger.error("Error setting up depth model: %s", e)
            self._capabilities = BackendCapabilities(supports_vlm=False, supports_depth=False)

    def _teardown(self) -> None:
        for attr in (
            "_yolo_configured_infer_model",
            "_yolo_infer_model",
            "_fastdepth_configured_infer_model",
            "_fastdepth_infer_model",
        ):
            try:
                obj = getattr(self, attr, None)
                if obj is None:
                    continue
                del obj
                setattr(self, attr, None)
            except Exception:
                logger.exception("Error releasing %s", attr)

        try:
            if self._device is not None and self._owns_device:
                self._device.release()
        except Exception:
            logger.exception("Error releasing VDevice")
        finally:
            # Always drop our reference. For shared devices this does NOT
            # release the underlying VDevice — main() owns and releases it.
            self._device = None

    # ------------------------------------------------------------------ #
    # Background thread
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        _set_thread_name("hailo")
        try:
            self._setup()
        except Exception:
            logger.exception("Hailo8Backend setup failed; thread exiting")
            return

        self._setup_complete.set()
        logger.info(
            "Hailo8Backend setup complete — capabilities: vlm=%s depth=%s",
            self._capabilities.supports_vlm,
            self._capabilities.supports_depth,
        )
        self._compute_reference_depths()

        while not self._stop_event.is_set():
            frame = self._capture.get()
            if frame is None:
                time.sleep(self.IDLE_SLEEP)
                continue
            try:
                result = self._process(frame)
                self._publish(result)
            except Exception:
                logger.exception("Hailo8Backend processing error")
                time.sleep(self.IDLE_SLEEP)

    def _publish(self, result: HailoResult) -> None:
        with self._lock:
            self._latest = result

    def _process(self, frame: np.ndarray) -> HailoResult:
        yolo_result = self._run_yolo(frame)

        depth_map = None
        with self._depth_lock:
            run_depth = self._depth_enabled
        if self._capabilities.supports_depth and (run_depth or not self._reference_depths_ready.is_set()):
            try:
                depth_map = self._run_depth(frame)
            except Exception:
                logger.exception("Depth estimation failed")

        return HailoResult(yolo_result=yolo_result, depth_map=depth_map, vlm_answer=None)

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #

    def _run_yolo(self, image: np.ndarray) -> YoloResult:
        original_h, original_w = image.shape[:2]
        rescaled_img, top, bottom, left, right = letterbox(image, self._desired_h, self._desired_w)

        input_data = np.ascontiguousarray(rescaled_img, dtype=np.uint8)
        bindings = self._yolo_configured_infer_model.create_bindings()
        bindings.input(self._yolo_infer_model.input_names[0]).set_buffer(input_data)
        for out_name in self._yolo_infer_model.output_names:
            out_shape = self._yolo_infer_model.output(out_name).shape
            bindings.output(out_name).set_buffer(np.empty(out_shape, dtype=np.float32))
        self._yolo_configured_infer_model.run([bindings], timeout=1000)
        output = bindings.output(out_name).get_buffer()

        detection_result = YoloResult(timestamp=datetime.now())
        for class_id, detections in enumerate(output):
            if class_id not in self._class_ids:
                continue
            for det in detections:
                y1_n, x1_n, y2_n, x2_n, conf = det
                if conf < self._confidence_threshold:
                    continue
                x1, y1, x2, y2 = bbox_unmap(
                    x1_n, y1_n, x2_n, y2_n,
                    self._desired_h, self._desired_w,
                    top, bottom, left, right,
                    original_h, original_w,
                )
                detection_result.detections.append(YoloDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=class_id,
                    label=COCO_CLASSES[class_id],
                ))
        return detection_result

    def _run_depth(self, image: np.ndarray) -> np.ndarray:
        original_h, original_w = image.shape[:2]
        input_image, _, _, _, _ = letterbox(image, self._fastdepth_desired_h, self._fastdepth_desired_w)

        if self._fastdepth_c == 4 and input_image.shape[2] == 3:
            h_in, w_in, _ = input_image.shape
            padded = np.zeros((h_in, w_in, 4), dtype=np.uint8)
            padded[:, :, :3] = input_image
            input_image = padded

        input_data = np.ascontiguousarray(input_image, dtype=np.uint8)
        bindings = self._fastdepth_configured_infer_model.create_bindings()
        bindings.input(self._fastdepth_infer_model.input_names[0]).set_buffer(input_data)
        for out_name in self._fastdepth_infer_model.output_names:
            out_shape = self._fastdepth_infer_model.output(out_name).shape
            out_format = self._fastdepth_infer_model.output(out_name).format
            type_str = str(out_format.type)
            if "UINT16" in type_str:
                np_dtype = np.uint16
            elif "FLOAT32" in type_str:
                np_dtype = np.float32
            else:
                np_dtype = np.uint8
            bindings.output(out_name).set_buffer(np.empty(out_shape, dtype=np_dtype))
        self._fastdepth_configured_infer_model.run([bindings], timeout=1000)
        output = bindings.output(out_name).get_buffer()
        depth_map = output.squeeze()
        return cv2.resize(depth_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    def _compute_reference_depths(self) -> None:
        if self._reference_image is None or self._red_zones is None or not self._capabilities.supports_depth:
            self._reference_depths_ready.set()
            return
        try:
            logger.info("Computing reference depth map for %d red zones", len(self._red_zones))
            depth_map = self._run_depth(self._reference_image)
            if depth_map is None:
                self._reference_depths_ready.set()
                return
            for idx, rz in enumerate(self._red_zones):
                poly = zone_to_polygon(rz)
                x1, y1, x2, y2 = polygon_bounding_box(poly)
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(depth_map.shape[1], int(x2))
                y2 = min(depth_map.shape[0], int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = depth_map[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                shifted = pts - np.array([[x1, y1]], dtype=np.int32)
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                cv2.fillPoly(mask, [shifted], 1)
                if mask.sum() == 0:
                    continue
                self._reference_depths[idx] = float(np.median(crop[mask.astype(bool)]))
            logger.info("Reference depths: %s", self._reference_depths)
        except Exception:
            logger.exception("Failed to compute reference depth map")
        self._reference_depths_ready.set()
