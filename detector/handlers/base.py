"""Reusable base class for action handlers.

Each handler instance is bound to a single action_id (one row in the
`actions` table) and processes DetectionEvents on its own background
thread so the pipeline hot loop is never blocked on network/disk I/O.
The pattern mirrors SampleSaverHandler.
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detector.events import DetectionEvent

logger = logging.getLogger(__name__)


def _build_metadata(event: DetectionEvent) -> dict:
    """Build the metadata.json payload for an event. Mirrors the shape used
    by SampleSaverHandler._save() so downstream consumers can rely on a
    consistent schema regardless of which alert type produced the file."""
    zone_meta = None
    if event.zone:
        zone_meta = {
            k: v for k, v in event.zone.items()
            if not isinstance(v, (bytes, bytearray))
        }
    return {
        "trigger": event.trigger,
        "timestamp": event.timestamp.isoformat(),
        "scene_id": event.scene_id,
        "detected_class": event.detected_class,
        "vlm_prompt": event.vlm_prompt,
        "vlm_answer": event.vlm_answer,
        "zone": zone_meta,
        "is_global_prompt": event.is_global_prompt,
    }


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR image as JPEG bytes. Returns empty bytes on failure."""
    if image is None:
        return b""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


class BaseActionHandler:
    """Base class for action handlers.

    Subclasses override `_deliver(event, payload)` to actually send the
    payload somewhere. This base handles:
      - a bounded queue (drops on overflow with a warning)
      - a daemon worker thread
      - graceful stop() with a drain timeout
      - lazy payload construction (raw + annotated JPEG + metadata)
    """

    QUEUE_MAX = 64
    STOP_DRAIN_TIMEOUT = 3.0

    def __init__(self, action_id: str, action_name: str):
        self._action_id = action_id
        self._action_name = action_name
        self._queue: "queue.Queue[DetectionEvent]" = queue.Queue(maxsize=self.QUEUE_MAX)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=self._thread_name())
        self._thread.start()
        logger.info("%s started for action %s", self.__class__.__name__, self._action_id)

    def _thread_name(self) -> str:
        return f"action-{self._action_id[:8]}"

    def __call__(self, event: DetectionEvent) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning(
                "%s queue full - dropping event trigger=%s action=%s",
                self.__class__.__name__, event.trigger, self._action_id,
            )

    def stop(self) -> None:
        self._stop_event.set()
        # Wake the worker if it's blocked on get()
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        self._thread.join(timeout=self.STOP_DRAIN_TIMEOUT)
        if self._thread.is_alive():
            logger.warning(
                "%s worker thread did not exit within %.1fs",
                self.__class__.__name__, self.STOP_DRAIN_TIMEOUT,
            )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL or self._stop_event.is_set():
                return
            try:
                self._handle(item)
            except Exception:
                logger.exception(
                    "%s failed to handle event trigger=%s action=%s",
                    self.__class__.__name__, item.trigger, self._action_id,
                )

    def _handle(self, event: DetectionEvent) -> None:
        payload = {
            "raw_jpeg": encode_jpeg(event.raw_frame) if event.raw_frame is not None else b"",
            "annotated_jpeg": encode_jpeg(event.annotated_image),
            "metadata": _build_metadata(event),
        }
        self._deliver(event, payload)

    def _deliver(self, event: DetectionEvent, payload: dict) -> None:
        raise NotImplementedError


_SENTINEL = object()
