import ctypes
import logging
import os
import subprocess
import threading
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detector.capture import FrameCapture
from detector.inference.preprocessing import COCO_CLASSES, bbox_unmap, letterbox
from detector.inference.protocols import (
    BackendCapabilities,
    HailoResult,
    InferenceBackend,
    VlmRequest,
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


class Hailo10Backend(InferenceBackend):
    """Hailo-10H backend: YOLO object detection + SCDepth depth estimation + Qwen3 VLM."""

    RELOAD_SETTLE_SECONDS: float = 2.0
    SETUP_TIMEOUT: float = 60.0
    IDLE_SLEEP: float = 0.05
    DEFAULT_VLM_PROMPT = "Is the {class} attacking a plant?"
    GLOBAL_DESCRIPTION_PROMPT = "Describe what you see in this image in one sentence."

    def __init__(
        self,
        capture: FrameCapture,
        yolo_classes: list[str],
        hef_config: dict,
        reference_image: Optional[np.ndarray] = None,
        red_zones: Optional[list] = None,
    ):
        self._capture = capture
        self.yolo_classes = yolo_classes
        self._class_ids = [COCO_CLASSES.index(c) for c in yolo_classes]
        self._reference_image = reference_image
        self._red_zones = red_zones
        self._confidence_threshold = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4"))

        self._yolo_path = Path(hef_config["yolo"]["path"])
        self._depth_path = Path(hef_config["depth"]["path"]) if "depth" in hef_config else None
        self._vlm_path = Path(hef_config["vlm"]["path"]) if "vlm" in hef_config else None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[HailoResult] = None
        self._setup_complete = threading.Event()

        self._device = None
        self._yolo_infer_model = None
        self._yolo_configured_infer_model = None
        self._desired_h: Optional[int] = None
        self._desired_w: Optional[int] = None
        self._c: Optional[int] = None

        self._fastdepth_infer_model = None
        self._fastdepth_configured_infer_model = None
        self._fastdepth_desired_h: Optional[int] = None
        self._fastdepth_desired_w: Optional[int] = None
        self._fastdepth_c: Optional[int] = None

        self._depth_enabled = False
        self._depth_lock = threading.Lock()

        self._vlm = None
        self._vlm_request_lock = threading.Lock()
        self._vlm_request: Optional[VlmRequest] = None
        self._vlm_drop_count: int = 0

        self._reference_depths: dict[int, float] = {}
        self._reference_depths_ready = threading.Event()

        # Capabilities are updated dynamically in _setup() if VLM/depth fail to load.
        self._capabilities = BackendCapabilities(supports_vlm=True, supports_depth=True)

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def start(self) -> None:
        if self._thread is not None:
            logger.warning("Hailo10Backend already started")
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

    def request_vlm(
        self,
        frame: np.ndarray,
        zone: Optional[dict],
        detected_class: Optional[str],
        is_global: bool = False,
        global_prompt: Optional[str] = None,
    ) -> None:
        with self._vlm_request_lock:
            if self._vlm_request is not None:
                self._vlm_drop_count += 1
                logger.debug("VLM request dropped (%d total)", self._vlm_drop_count)
                return
            self._vlm_request = VlmRequest(
                frame=frame.copy(),
                zone=zone,
                detected_class=detected_class,
                is_global=is_global,
                global_prompt=global_prompt,
            )

    # ------------------------------------------------------------------ #
    # Setup / teardown
    # ------------------------------------------------------------------ #

    def _setup(self) -> None:
        from hailo_platform import Device, HailoSchedulingAlgorithm, VDevice

        device_ids = self._get_hailo_device_ids()
        if not device_ids:
            raise RuntimeError("No Hailo devices found")
        for dev_id in device_ids:
            logger.info(self._get_hailo_device_info(dev_id))

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self._device = VDevice(params)

        self._setup_yolo()
        self._setup_depth()
        self._setup_vlm()

    def _setup_yolo(self) -> None:
        try:
            self._yolo_infer_model = self._device.create_infer_model(str(self._yolo_path))
            input_name = self._yolo_infer_model.input_names[0]
            self._desired_h, self._desired_w, self._c = self._yolo_infer_model.input(input_name).shape
            self._yolo_configured_infer_model = self._yolo_infer_model.configure()
            logger.info("YOLO configured (%dx%d)", self._desired_h, self._desired_w)
        except Exception as e:
            logger.error("Error setting up YOLO: %s", e)
            raise

    def _setup_depth(self) -> None:
        if self._depth_path is None:
            logger.info("No depth HEF in manifest; depth disabled")
            self._capabilities = BackendCapabilities(
                supports_vlm=self._capabilities.supports_vlm,
                supports_depth=False,
            )
            return
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
            self._capabilities = BackendCapabilities(
                supports_vlm=self._capabilities.supports_vlm,
                supports_depth=False,
            )

    def _setup_vlm(self) -> None:
        if self._vlm_path is None:
            logger.info("No VLM HEF in manifest; VLM disabled")
            self._capabilities = BackendCapabilities(
                supports_vlm=False,
                supports_depth=self._capabilities.supports_depth,
            )
            return
        try:
            from hailo_platform.genai import VLM
            logger.info("Loading VLM from %s", self._vlm_path)
            self._vlm = VLM(self._device, str(self._vlm_path))
            logger.info("VLM configured")
        except Exception as e:
            logger.error("Error setting up VLM: %s", e)
            self._vlm = None
            self._capabilities = BackendCapabilities(
                supports_vlm=False,
                supports_depth=self._capabilities.supports_depth,
            )

    def _teardown(self) -> None:
        for attr in (
            "_vlm",
            "_yolo_configured_infer_model",
            "_yolo_infer_model",
            "_fastdepth_configured_infer_model",
            "_fastdepth_infer_model",
        ):
            try:
                obj = getattr(self, attr, None)
                if obj is None:
                    continue
                if attr == "_vlm" and hasattr(obj, "release"):
                    try:
                        obj.release()
                    except Exception:
                        logger.debug("Ignored error releasing %s", attr, exc_info=True)
                del obj
                setattr(self, attr, None)
            except Exception:
                logger.exception("Error releasing %s", attr)

        try:
            if self._device is not None:
                self._device.release()
                self._device = None
        except Exception:
            logger.exception("Error releasing VDevice")

    # ------------------------------------------------------------------ #
    # Background thread
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        _set_thread_name("hailo")
        try:
            self._setup()
        except Exception:
            logger.exception("Hailo10Backend setup failed; thread exiting")
            return

        self._setup_complete.set()
        logger.info(
            "Hailo10Backend setup complete — capabilities: vlm=%s depth=%s",
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
                logger.exception("Hailo10Backend processing error")
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

        vlm_answer = None
        with self._vlm_request_lock:
            request = self._vlm_request
            self._vlm_request = None
        if request is not None and self._vlm is not None:
            try:
                prompt = self._resolve_prompt(request)
                logger.info("Running VLM — prompt: %s", prompt)
                vlm_answer = self._run_vlm(request.frame, prompt, is_global=request.is_global)
                logger.info("VLM answer: %s", vlm_answer)
            except Exception:
                logger.exception("VLM inference failed")

        return HailoResult(yolo_result=yolo_result, depth_map=depth_map, vlm_answer=vlm_answer)

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

    def _run_vlm(self, frame: np.ndarray, question: str, is_global: bool = False) -> str:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 288)).astype(np.uint8)

        if is_global:
            system_msg = (
                "You are a visual analyst. Describe only what you actually see. "
                "Be brief, don't use too many adjectives"
            )
            max_tokens = 200
        else:
            system_msg = (
                "You are a visual analyst. Look carefully at the image. "
                "Answer the question with Yes or No as your very first word, "
                "then optionally explain briefly."
            )
            max_tokens = 20

        prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ]},
        ]

        self._vlm.clear_context()
        try:
            response = self._vlm.generate_all(
                prompt=prompt,
                frames=[image],
                temperature=0.1,
                max_generated_tokens=max_tokens,
            )
        except Exception:
            self._vlm.clear_context()
            raise

        answer_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        answer_text = answer_text.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()

        if is_global:
            logger.info("VLM description: %r", answer_text[:120])
            return answer_text

        first_word = answer_text.split()[0].lower().rstrip(".,;:!?") if answer_text.split() else ""
        logger.info("VLM response=%r -> %s", answer_text[:60], first_word)
        return "Yes" if first_word == "yes" else "No"

    def _resolve_prompt(self, request: VlmRequest) -> str:
        if request.is_global:
            return self.GLOBAL_DESCRIPTION_PROMPT
        template = (request.zone.get("vlm_prompt") if request.zone else None) or self.DEFAULT_VLM_PROMPT
        return template.replace("{class}", request.detected_class or "") + " Answer with only 'Yes' or 'No':"

    # ------------------------------------------------------------------ #
    # Reference depth computation
    # ------------------------------------------------------------------ #

    def _compute_reference_depths(self) -> None:
        if self._reference_image is None or self._red_zones is None or not self._capabilities.supports_depth:
            self._reference_depths_ready.set()
            return
        try:
            logger.info("Computing reference depth map for %d red zones", len(self._red_zones))
            depth_map = self._run_depth(self._reference_image)
            if depth_map is None:
                logger.warning("Reference depth map is None; skipping per-zone depths")
                self._reference_depths_ready.set()
                return
            for idx, rz in enumerate(self._red_zones):
                x = max(0, int(rz["x"]))
                y = max(0, int(rz["y"]))
                w = int(rz["width"])
                h = int(rz["height"])
                x2 = min(depth_map.shape[1], x + w)
                y2 = min(depth_map.shape[0], y + h)
                if x2 <= x or y2 <= y:
                    logger.warning("Zone %d has zero-area crop; skipping reference depth", idx)
                    continue
                crop = depth_map[y:y2, x:x2]
                if crop.size > 0:
                    self._reference_depths[idx] = float(np.median(crop))
            logger.info("Reference depths: %s", self._reference_depths)
        except Exception:
            logger.exception("Failed to compute reference depth map")
        self._reference_depths_ready.set()

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_hailo_device_ids() -> list:
        from hailo_platform import Device
        with Device() as dev:
            ids = dev.scan()
            logger.info("Available Hailo devices: %s", ids)
        return ids

    @staticmethod
    def _get_hailo_device_info(device_id: str) -> str:
        try:
            result = subprocess.run(
                ["lspci", "-v", "-s", device_id],
                capture_output=True, text=True, check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
