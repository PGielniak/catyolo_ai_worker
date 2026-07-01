from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
    frame: np.ndarray
    zone: Optional[dict]
    detected_class: Optional[str]
    is_global: bool = False
    global_prompt: Optional[str] = None


@dataclass
class BackendCapabilities:
    supports_vlm: bool
    supports_depth: bool
    max_concurrent_streams: int = 1


class InferenceBackend(ABC):
    """Hardware-neutral interface for a Hailo inference backend.

    Implementations: Hailo10Backend (YOLO + depth + VLM),
                     Hailo8Backend  (YOLO + optional depth, no VLM).
    """

    # Subclasses override these to communicate device-specific timing to
    # DetectionPipeline.reload_config().
    RELOAD_SETTLE_SECONDS: float = 2.0
    SETUP_TIMEOUT: float = 60.0

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self, timeout: float = 2.0) -> None: ...

    @abstractmethod
    def wait_until_ready(self, timeout: float = 30.0) -> bool: ...

    @abstractmethod
    def get_latest(self) -> Optional[HailoResult]: ...

    @abstractmethod
    def set_depth_enabled(self, enabled: bool) -> None: ...

    @abstractmethod
    def get_reference_depths(self, timeout: float = 0.0) -> tuple[bool, dict[int, float]]: ...

    def get_depth_tuning(self) -> dict:
        """Live depth-pipeline tuning values. Default: empty (unsupported)."""
        return {}

    def set_depth_tuning(self, params: dict) -> dict:
        """Update live depth-pipeline tuning values. Default: no-op."""
        return {}

    def request_vlm(
        self,
        frame: np.ndarray,
        zone: Optional[dict],
        detected_class: Optional[str],
        is_global: bool = False,
        global_prompt: Optional[str] = None,
    ) -> None:
        """Queue a VLM inference request. Default no-op for non-VLM backends."""

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...
