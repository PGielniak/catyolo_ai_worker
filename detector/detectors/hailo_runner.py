# Deprecated — use detector.inference.hailo10_backend.Hailo10Backend directly.
# This shim exists so external scripts importing HailoRunner by name continue to work.
from detector.inference.hailo10_backend import Hailo10Backend as HailoRunner  # noqa: F401
from detector.inference.protocols import (  # noqa: F401
    HailoResult,
    VlmRequest,
    YoloDetection,
    YoloResult,
)
