import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
import hailo_platform
import contextlib
from hailo_platform import Device, VDevice, HailoSchedulingAlgorithm
from hailo_platform.genai import VLM 
import numpy as np
import cv2
import os
import subprocess
from datetime import datetime
from typing import Optional
from detector.detectors.base import BaseDetector
from detector.capture import FrameCapture
import time

import ctypes

def _set_thread_name(name):
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass


logger = logging.getLogger(__name__)

@dataclass
class YoloDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str


@dataclass
class YoloResult:
    timestamp: float
    detections: list = field(default_factory=list)           

@dataclass
class HailoResult:
    yolo_result: Optional[YoloResult] = None
    depth_map: Optional[np.ndarray] = None
    vlm_answer: Optional[str] = None


@dataclass
class VlmRequest:
    """Overlap data handed from the pipeline to the Hailo runner so the runner can
    build and run the proper VLM call (overlap detection itself stays in the pipeline)."""
    frame: np.ndarray
    zone: Optional[dict]
    detected_class: Optional[str]
    is_global: bool = False
    global_prompt: Optional[str] = None


class HailoRunner(BaseDetector):
    YOLO_PATH = Path("/mnt/ssd/home/patryk/pycharm/catyolo_ai_worker/hefs/yolov11x.hef")
    DEPTH_PATH = Path("/mnt/ssd/home/patryk/pycharm/catyolo_ai_worker/hefs/scdepthv3.hef")
    VLM_PATH = Path("/mnt/ssd/hailo-ollama-models/Qwen3-VL-2B-Instruct.hef")
    DEFAULT_VLM_PROMPT = "Is the {class} attacking a plant?"
    IDLE_SLEEP = 0.05
    COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]
    def __init__(self, capture: FrameCapture, yolo_classes: list[str],
                 reference_image: Optional[np.ndarray] = None, red_zones: Optional[list] = None):
        self._capture = capture

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[HailoResult] = None

        self.yolo_classes = yolo_classes
        self._class_ids = [self.COCO_CLASSES.index(c) for c in yolo_classes]
        self._setup_complete = threading.Event()
        self._device = None
        self._yolo_infer_model = None
        self._yolo_configured_infer_model = None
        self._desired_h = None
        self._desired_w = None
        self._c = None
        self._depth_enabled = False
        self._depth_lock = threading.Lock()

        self._fastdepth_infer_model = None
        self._fastdepth_configured_infer_model = None

        self._vlm = None
        self._vlm_request_lock = threading.Lock()
        self._vlm_request: Optional[VlmRequest] = None

        self._reference_image = reference_image
        self._red_zones = red_zones
        self._reference_depths: dict[int, float] = {}
        self._reference_depths_ready = threading.Event()



    def start(self):
        if self._thread is not None:
            logger.warning("Hailo Pipeline already started")
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="occlusion"
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._teardown() 

    def _setup(self):
        device_ids = self.get_hailo_device_ids()
        if len(device_ids) < 1:
            message = "No Hailo devices found"
            logger.error(message)
            raise RuntimeError(message)
        for id in device_ids:
            device_info = self.get_hailo_device_info(id)
            logger.info(device_info)

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self._device = VDevice(params)

        try:
            self._yolo_infer_model = self._device.create_infer_model(str(self.YOLO_PATH))
            input_name = self._yolo_infer_model.input_names[0]
            desired_h, desired_w, c = self._yolo_infer_model .input(input_name).shape
            self._desired_h = desired_h
            self._desired_w = desired_w
            self._c = c
            for i in self._yolo_infer_model.input_names:
                logger.debug(f"input_name {i}")
                logger.debug(repr(self._yolo_infer_model.input(i)))
            logger.info(type(self._yolo_infer_model ))
            h, w, c = self._yolo_infer_model.input(input_name).shape
            self._yolo_configured_infer_model = self._yolo_infer_model.configure()
            logger.info("Yolo configured")
        except Exception as e:
            logger.error(f"Error while creating yolo infer model: {e}")
            raise e

        try:
            self._fastdepth_infer_model = self._device.create_infer_model(str(self.DEPTH_PATH))
            input_name = self._fastdepth_infer_model.input_names[0]

            for i in self._fastdepth_infer_model.input_names:
                print(f"input_name {i}")
                print(repr(self._fastdepth_infer_model.input(i)))
            print(type(self._fastdepth_infer_model))
            self._fastdepth_desired_h, self._fastdepth_desired_w, self._fastdepth_c = self._fastdepth_infer_model.input(input_name).shape
            self._fastdepth_configured_infer_model = self._fastdepth_infer_model.configure()
            logger.info("Depth configured")
        except Exception as e:
            logger.error(f"Error while creating fast depth infer model: {e}")
            raise e

        try:
            logger.info(f"Loading VLM from {self.VLM_PATH}")
            self._vlm = VLM(self._device, str(self.VLM_PATH))
            logger.info("VLM configured")
        except Exception as e:
            logger.error(f"Error while creating VLM: {e}")
            self._vlm = None



    def set_depth_enabled(self, enabled: bool):
        with self._depth_lock:
            self._depth_enabled = enabled

    def get_reference_depths(self, timeout: float = 60.0) -> tuple[bool, dict[int, float]]:
        ready = self._reference_depths_ready.wait(timeout=timeout)
        with self._lock:
            return ready, dict(self._reference_depths)

    def _compute_reference_depths(self):
        if self._reference_image is None or self._red_zones is None:
            self._reference_depths_ready.set()
            return
        try:
            logger.info("Computing reference depth map for %d red zones", len(self._red_zones))
            depth_map = self._run_depth_estimation(self._reference_image)
            if depth_map is None:
                logger.warning("Reference depth map is None, skipping per-zone reference depths")
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
                    logger.warning("Zone %d has zero-area crop, skipping reference depth", idx)
                    continue
                crop = depth_map[y:y2, x:x2]
                if crop.size > 0:
                    self._reference_depths[idx] = float(np.median(crop))
            logger.info("Reference depths computed: %s", self._reference_depths)
        except Exception:
            logger.exception("Failed to compute reference depth map")
        self._reference_depths_ready.set()

    def request_vlm(self, frame: np.ndarray, zone: Optional[dict], detected_class: Optional[str],
                    is_global: bool = False, global_prompt: Optional[str] = None):
        """Queue a VLM call. For zone-based requests, zone and detected_class are required.
        For global prompt requests, set is_global=True and pass global_prompt."""
        with self._vlm_request_lock:
            if self._vlm_request is not None:
                return
            self._vlm_request = VlmRequest(
                frame=frame.copy(),
                zone=zone,
                detected_class=detected_class,
                is_global=is_global,
                global_prompt=global_prompt,
            )

    GLOBAL_DESCRIPTION_PROMPT = "Describe what you see in this image in one sentence."

    def _resolve_prompt(self, request: VlmRequest) -> str:
        if request.is_global:
            return self.GLOBAL_DESCRIPTION_PROMPT
        prompt_template = (request.zone.get("vlm_prompt") if request.zone else None) or self.DEFAULT_VLM_PROMPT
        question = prompt_template.replace("{class}", request.detected_class or "")
        return question + "Answer with only 'Yes' or 'No':"
    def _teardown(self):
        """Release all infer models and the VDevice. Must be called only after
        the runner's thread has fully exited (via stop()), and callers should
        give HailoRT a brief settle delay afterwards before opening a new
        VDevice — see DetectionPipeline.reload_config()."""
        # Order matters: release sub-models first, then the VDevice.
        # Releasing the VDevice while a sub-model is still in use produces
        # HAILO_STREAM_NOT_ACTIVATED(72) and "Lost communication with the server".
        for attr in (
            "_vlm",
            "_yolo_configured_infer_model",
            "_yolo_infer_model",
            "_fastdepth_configured_infer_model",
            "_fastdepth_infer_model",
        ):
            try:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    obj = getattr(self, attr)
                    # InferModel / ConfiguredInferModel don't expose release();
                    # just del the reference and let the C++ object destruct.
                    if attr == "_vlm" and hasattr(obj, "release"):
                        try:
                            obj.release()
                        except Exception:
                            logger.debug(f"Ignored error releasing {attr}", exc_info=True)
                    del obj
                    setattr(self, attr, None)
            except Exception:
                logger.exception(f"Error releasing {attr}")

        try:
            if hasattr(self, "_device") and self._device is not None:
                self._device.release()
                self._device = None
        except Exception:
            logger.exception("Error releasing VDevice")


        
    def get_latest(self) -> Optional[HailoResult]:
        with self._lock:
            return self._latest  

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        return self._setup_complete.wait(timeout=timeout)

    def get_hailo_device_ids(self):
        with Device() as dev:
            device_ids = dev.scan()
            logger.info("Available devices:", device_ids)
            ids = device_ids
            logger.info(ids)
            logger.info(f"Device: {dev}")
        return ids

    def get_hailo_device_info(self, device_id="0001:01:00.0"):
        try:
            command = ["lspci", "-v", "-s", device_id]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"

    def _rescale_image(self, original_image: np.ndarray, desired_size: tuple[int, int]) -> tuple[np.ndarray,int,int,int,int]:
        original_height, original_width = original_image.shape[:2]
        desired_height, desired_width = desired_size

        scaling_factor = min(desired_height / original_height, desired_width / original_width)
        new_h, new_w = int(original_height * scaling_factor), int(original_width * scaling_factor)

        resized = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_x = desired_height - resized.shape[0]
        pad_y = desired_width - resized.shape[1]

        logger.debug(f"pad x {pad_x}")
        logger.debug(f"pad y {pad_y}")
        
        top =  pad_x // 2
        bottom = (pad_x) - top
        left = pad_y // 2
        right = (pad_y) - left

        logger.debug(f"top pad {top}")
        logger.debug(f"bottom pad {bottom}")
        logger.debug(f"left pad {left}")
        logger.debug(f"right_pad {right}")
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return tuple([padded, top, bottom, left, right])


    def _run_yolo_object_detection(self, image: np.ndarray) -> HailoResult:
        original_h, original_w = image.shape[:2]

        
        logger.debug(f"Original image shape: {original_h}x{original_w}")
        logger.debug(f"Desired image shape: {self._desired_h}x{self._desired_w}")

        rescaled_img, top, bottom, left, right = self._rescale_image(image, (self._desired_h, self._desired_w))
        logger.debug(f"Rescaled image shape: {rescaled_img.shape}")
        logger.debug(f"Rescaled image shape: {self._desired_h}x{self._desired_w}")

        input_data = np.ascontiguousarray(rescaled_img, dtype=np.uint8)
        bindings = self._yolo_configured_infer_model.create_bindings()
        bindings.input(self._yolo_infer_model .input_names[0]).set_buffer(input_data)
        for out_name in self._yolo_infer_model.output_names:
            out_shape = self._yolo_infer_model.output(out_name).shape
            bindings.output(out_name).set_buffer(np.empty(out_shape, dtype=np.float32))
        self._yolo_configured_infer_model.run([bindings], timeout=1000)
        output = bindings.output(out_name).get_buffer()
        logger.debug(output)

        confidence_threshold = 0.05
        results = []
        detection_result = YoloResult(
                timestamp=datetime.now(),
            )
        for class_id, detections in enumerate(output):
            if class_id not in self._class_ids:
                continue

            logger.debug(f"Class to detect {[class_id for class_id in self._class_ids]}\n")
            for det in detections:
                y1_n, x1_n, y2_n, x2_n, conf = det
                if conf < confidence_threshold:
                    continue

                offset = 0

                x1 = int(x1_n * self._desired_h - offset)
                y1 = int(y1_n * self._desired_w + offset)
                x2 = int(x2_n * self._desired_h - offset)
                y2 = int(y2_n * self._desired_w + offset)

                x1 -= left
                y1 -= top
                x2 -= left
                y2 -= top

                x1 = int(x1 * original_w / (self._desired_w - left - right))
                y1 = int(y1 * original_h / (self._desired_h - top - bottom))
                x2 = int(x2 * original_w / (self._desired_w - left - right))
                y2 = int(y2 * original_h / (self._desired_h - top - bottom))


                detection = YoloDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=class_id,
                    label=self.COCO_CLASSES[class_id]
                )

                detection_result.detections.append(detection)

        return detection_result

    def _run(self):
        _set_thread_name("hailo") 
        try:
            self._setup()
        except Exception:
            logger.exception("HailoPipeline setup failed; thread exiting")
            return
        
        self._setup_complete.set()
        self._compute_reference_depths()
        logger.info("HailoPipeline running")
        
        while not self._stop_event.is_set():
            frame = self._capture.get()
            if frame is None:
                time.sleep(self.IDLE_SLEEP)
                continue
            
            try:
                result = self._process(frame)
                self._publish(result)
            except Exception:
                logger.exception("Hailo Pipeline processing error")
                time.sleep(self.IDLE_SLEEP)
    
    def _publish(self, result: HailoResult):
        with self._lock:
            self._latest = result


    def _process(self, frame: np.ndarray) -> HailoResult:
        yolo_detections = self._run_yolo_object_detection(image=frame)

        depth_map = None
        with self._depth_lock:
            run_depth = self._depth_enabled
        if run_depth or not self._reference_depths_ready.is_set():
            try:
                depth_map = self._run_depth_estimation(image=frame)
            except Exception:
                logger.exception("Depth estimation failed")

        vlm_answer = None
        with self._vlm_request_lock:
            request = self._vlm_request
            self._vlm_request = None
        if request is not None and self._vlm is not None:
            try:
                prompt = self._resolve_prompt(request)
                logger.info(f"Running VLM — prompt: {prompt}")
                vlm_answer = self._run_vlm(request.frame, prompt, is_global=request.is_global)
                logger.info(f"VLM answer: {vlm_answer}")
            except Exception:
                logger.exception("VLM inference failed")

        return HailoResult(
            yolo_result=yolo_detections,
            depth_map=depth_map,
            vlm_answer=vlm_answer,
        )

    def _run_vlm(self, frame: np.ndarray, question: str, is_global: bool = False) -> str:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Qwen3-VL-2B expects 288x512 (HxW); Qwen2 used 336x336
        image = cv2.resize(image, (512, 288)).astype(np.uint8)

        if is_global:
            system_msg = "You are a visual analyst. Describe only what you actually see. Be brief, don't use too many adjectives"
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
                {"type": "image"}
            ]}
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

        import re
        answer_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        answer_text = answer_text.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()

        if is_global:
            logger.info(f"VLM description: {repr(answer_text[:120])}")
            return answer_text

        first_word = answer_text.split()[0].lower().rstrip(".,;:!?") if answer_text.split() else ""
        logger.info(f"VLM response={repr(answer_text[:60])} -> {first_word}")
        if first_word == "yes":
            return "Yes"
        return "No"

    def _resize_for_depth(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        desired_h = self._fastdepth_desired_h
        desired_w = self._fastdepth_desired_w
        scale = min(desired_h / h, desired_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = (desired_h - new_h) // 2
        left = (desired_w - new_w) // 2
        padded = cv2.copyMakeBorder(
            resized, top, desired_h - new_h - top, left, desired_w - new_w - left,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded

    def _run_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        original_h, original_w = image.shape[:2]
        input_image = self._resize_for_depth(image)
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
            if "UINT8" in type_str:
                np_dtype = np.uint8
            elif "UINT16" in type_str:
                np_dtype = np.uint16
            elif "FLOAT32" in type_str:
                np_dtype = np.float32
            else:
                np_dtype = np.uint8
            bindings.output(out_name).set_buffer(np.empty(out_shape, dtype=np_dtype))
        self._fastdepth_configured_infer_model.run([bindings], timeout=1000)
        output = bindings.output(out_name).get_buffer()
        depth_map = output.squeeze()
        depth_map = cv2.resize(depth_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return depth_map
