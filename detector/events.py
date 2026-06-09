import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    annotated_image: np.ndarray
    trigger: str
    timestamp: datetime
    raw_frame: Optional[np.ndarray] = None
    zone: Optional[dict] = None
    detected_class: Optional[str] = None
    vlm_prompt: Optional[str] = None
    vlm_answer: Optional[str] = None


# Kept for backward compatibility with any external subscribers.
VlmEvent = DetectionEvent


class DetectionEventEmitter:
    def __init__(self):
        self._handlers: list[Callable[[DetectionEvent], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, handler: Callable[[DetectionEvent], None]):
        with self._lock:
            self._handlers.append(handler)

    def emit(self, event: DetectionEvent):
        with self._lock:
            handlers = list(self._handlers)
        logger.info(
            "DetectionEvent emitted — trigger=%s class=%s zone=%s handlers=%d prompt=%s answer=%s",
            event.trigger,
            event.detected_class or "-",
            "global" if event.zone is None else event.zone.get("id", "?"),
            len(handlers),
            (event.vlm_prompt or "")[:80],
            event.vlm_answer or "-",
        )
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("DetectionEvent handler failed")


# Kept for backward compatibility with any external subscribers.
VlmEventEmitter = DetectionEventEmitter
