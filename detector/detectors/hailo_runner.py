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
    """Set the OS-visible thread name (Linux only, max 15 chars)."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)  # 15 = PR_SET_NAME
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

class HailoRunner(BaseDetector):
    """
    Runs Hailo inference in sequence on a separate thread
    """
    YOLO_PATH = Path("/mnt/ssd/home/patryk/pycharm/catyolo_ai_worker/hefs/yolov11x.hef")
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
    def __init__(self, capture: FrameCapture, yolo_classes: list[str]):
        self._capture = capture

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[HailoResult] = None

        self.yolo_classes = yolo_classes
        self._class_ids = [self.COCO_CLASSES.index(c) for c in yolo_classes]
        # self._yolo = YoloDetection()
        self._setup_complete = threading.Event()
        self._device = None
        self._yolo_infer_model = None
        self._yolo_configured_infer_model = None
        self._desired_h = None
        self._desired_w = None
        self._c = None



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

    def _setup(self):
        """Load models, fetch config, etc. Called once before loop."""
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
        params.group_id = "SHARED"  # NOT multi_process_service on Hailo10H
        self._device = VDevice(params)
        self._yolo_infer_model = self._device.create_infer_model(str(self.YOLO_PATH))
        input_name = self._yolo_infer_model .input_names[0]
        desired_h, desired_w, c = self._yolo_infer_model .input(input_name).shape
        self._desired_h = desired_h
        self._desired_w = desired_w
        self._c = c
        for i in self._yolo_infer_model .input_names:
            logger.debug(f"input_name {i}")
            logger.debug(repr(self._yolo_infer_model .input(i)))
        logger.info(type(self._yolo_infer_model ))
        h, w, c = self._yolo_infer_model .input(input_name).shape
        self._yolo_configured_infer_model = self._yolo_infer_model .configure()


    def _teardown(self):
        """Release device on the same thread that created it."""
        try:
            if self._device is not None:
                self._device.release()
        except Exception:
            logger.exception("error during Hailo teardown")

        
    def get_latest(self) -> Optional[HailoResult]:
        """Return the most recent result, or None if no frame has been processed yet."""
        with self._lock:
            return self._latest  

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until Hailo setup completes. Returns False on timeout."""
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
        """
        Retrieve detailed information about a Hailo device using the `lspci` command.

        Args:
            device_id (str): The BDF identifier of the Hailo device (e.g., "0001:01:00.0").

        Returns:
            str: The output of the `lspci` command for the specified device.
        """
        try:
            # Run the `lspci` command and capture the output
            command = ["lspci", "-v", "-s", device_id]
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            # Return the output
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"

    def _rescale_image(self, original_image: np.ndarray, desired_size: tuple[int, int]) -> tuple[np.ndarray,int,int,int,int]:
        """
        Rescale the image to the desired size while maintaining the aspect ratio.
        """
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

                offset = 0 # 172

                x1 = int(x1_n * self._desired_h - offset)   # 0.5 * 640 = 320px
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
        """Run Yolo Detection on a frame"""

        yolo_detections = self._run_yolo_object_detection(image=frame)

        return HailoResult(
            yolo_result=yolo_detections
        )


