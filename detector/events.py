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
    # True for events produced by the global scene-prompt timer (zone=None,
    # trigger="global_description"). Action handlers use this to look up the
    # scene-level scene_prompt_action_ids instead of the per-zone action_ids.
    is_global_prompt: bool = False
    # The scene this event originated from (multi-camera). Set by the
    # pipeline from its SceneConfig so every downstream consumer (sample
    # folders, telegram caption, webhook metadata, SMB folder, logs) can
    # disambiguate events across scenes.
    scene_id: Optional[str] = None


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
            "DetectionEvent emitted — scene=%s trigger=%s class=%s zone=%s handlers=%d prompt=%s answer=%s",
            event.scene_id or "-",
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


# ── Dispatcher interface ──────────────────────────────────────────────────────
# The pipeline calls this AFTER emitting on the in-process bus. It's a
# separate channel so the dispatch registry (which talks to SMB / Telegram /
# the public internet) doesn't have to subscribe to every event just to filter
# by action_id — the pipeline has already resolved the zone's action_ids and
# passes them along.

DispatcherFn = Callable[[DetectionEvent, list[str]], None]
"""Called with the event and the list of action IDs that should fire for it.

For zone events, this is `event.zone["action_ids"]`. For global-prompt events,
this is the scene's `scene_prompt_action_ids`. The dispatcher is responsible
for looking up the action configs (by ID), filtering to types it knows how
to handle, and fanning the event out to the right handler instances.
"""


class _DispatcherHolder:
    """Mutable singleton holder so pipeline.py doesn't need to know about
    the registry module. Set via DetectionEvent.set_dispatcher(...)."""

    def __init__(self):
        self.fn: Optional[DispatcherFn] = None


_dispatcher = _DispatcherHolder()


def set_dispatcher(fn: Optional[DispatcherFn]) -> None:
    """Install (or clear with None) the pipeline-level dispatcher.

    Called once at worker startup. The dispatcher is invoked after the
    in-process event bus so existing SampleSaverHandler subscribers keep
    working unchanged.
    """
    _dispatcher.fn = fn


def get_dispatcher() -> Optional[DispatcherFn]:
    return _dispatcher.fn


def dispatch_event(event: DetectionEvent, action_ids: list[str]) -> None:
    """Invoke the registered dispatcher (if any) with the given event and
    action_ids. No-op when no dispatcher is registered. Exceptions raised by
    the dispatcher are logged and swallowed — never propagate into the
    pipeline hot loop.
    """
    fn = _dispatcher.fn
    if fn is None:
        return
    try:
        fn(event, action_ids)
    except Exception:
        logger.exception("DetectionEvent dispatcher raised; ignoring")

