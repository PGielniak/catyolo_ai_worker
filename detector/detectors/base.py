# detector/detectors/base.py
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    timestamp: float
    data: Any  # subclasses define structure
    
    
class BaseDetector(ABC):
    """Runs inference in a background thread, exposes latest result."""
    
    def __init__(self, capture, name: str):
        self._capture = capture
        self.name = name
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[DetectionResult] = None
        self._last_frame_id = None
    
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=self.name)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def get_latest(self) -> Optional[DetectionResult]:
        with self._lock:
            return self._latest  # immutable read; safe to share if subclasses don't mutate
    
    def _publish(self, data: Any):
        result = DetectionResult(timestamp=time.time(), data=data)
        with self._lock:
            self._latest = result
    
    @abstractmethod
    def _setup(self):
        """Load models, fetch config, etc. Called once before loop."""
        pass
    
    @abstractmethod
    def _process(self, frame) -> Any:
        """Process a single frame, return result data."""
        pass
    
    def _run(self):
        try:
            self._setup()
        except Exception as e:
            logger.exception(f"{self.name} setup failed: %s", e)
            return
        
        while not self._stop_event.is_set():
            frame = self._capture.get()
            if frame is None:
                time.sleep(0.05)
                continue
            
            try:
                data = self._process(frame)
                self._publish(data)
            except Exception:
                logger.exception(f"{self.name} processing error")
                time.sleep(0.1)