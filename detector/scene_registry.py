"""Per-scene pipeline registry (multi-camera support, WS2).

Owns one FrameCapture + one DetectionPipeline per scene. All scenes share:
  - one Hailo VDevice (injected as `shared_device`), multiplexed by the
    HailoRT ROUND_ROBIN scheduler
  - one ActionHandlerRegistry (the dispatcher is a process-wide singleton set
    in main(); events carry scene_id so handlers disambiguate)

The ConfigWatcher calls `registry.apply(changed, removed)` with the per-scene
diff; the registry reconciles by starting / reloading / stopping pipelines.
Mirrors the diff-and-reconcile pattern proven in ActionHandlerRegistry.
"""

import logging
import threading
from typing import Any, Callable, Optional

from detector.capture import FrameCapture
from detector.config import SceneConfig
from detector.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)


class ScenePipelineRegistry:
    """Lifecycle owner of one (FrameCapture, DetectionPipeline) per scene_id."""

    def __init__(
        self,
        api_base: str,
        shared_device: Any,
        max_scenes: int = 3,
    ):
        self._api_base = api_base
        self._shared_device = shared_device
        self._max_scenes = max_scenes
        # scene_id -> {"capture", "pipeline", "rtsp_url"}
        self._entries: dict[str, dict] = {}
        self._lock = threading.Lock()
        # Handlers subscribed to EVERY pipeline's event emitter (existing + new).
        self._global_subscribers: list[Callable] = []
        self._stopped = False

    # ------------------------------------------------------------------ #
    # Subscriptions
    # ------------------------------------------------------------------ #

    def subscribe(self, handler: Callable) -> None:
        """Subscribe a handler to every pipeline's event emitter.

        New pipelines started after this call are also subscribed. Used by the
        SampleSaverHandler so one saver instance covers all scenes.
        """
        with self._lock:
            self._global_subscribers.append(handler)
            entries = list(self._entries.values())
        for entry in entries:
            entry["pipeline"].subscribe(handler)

    # ------------------------------------------------------------------ #
    # Reconcile (driven by ConfigWatcher)
    # ------------------------------------------------------------------ #

    def apply(self, changed: list[SceneConfig], removed: list[str]) -> None:
        """Reconcile the running scene set with the backend's current state.

        Stops pipelines for removed scene_ids, reloads changed scenes, and
        starts new ones (subject to the max_scenes cap).
        """
        for scene_id in removed or []:
            self.stop(scene_id)
        for cfg in changed or []:
            scene_id = cfg.scene.get("scene_id")
            if not scene_id:
                logger.warning("Scene config has no scene_id; skipping")
                continue
            if self.has(scene_id):
                self.reload(cfg)
            else:
                self.start(cfg)

    # ------------------------------------------------------------------ #
    # Per-scene lifecycle
    # ------------------------------------------------------------------ #

    def has(self, scene_id: str) -> bool:
        with self._lock:
            return scene_id in self._entries

    def scene_ids(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    def get(self, scene_id: str) -> Optional[tuple[DetectionPipeline, FrameCapture]]:
        with self._lock:
            entry = self._entries.get(scene_id)
            if entry is None:
                return None
            return entry["pipeline"], entry["capture"]

    def items(self) -> list[tuple[str, DetectionPipeline, FrameCapture]]:
        with self._lock:
            return [
                (sid, e["pipeline"], e["capture"])
                for sid, e in self._entries.items()
            ]

    def start(self, cfg: SceneConfig) -> None:
        scene_id = cfg.scene.get("scene_id")
        if not scene_id:
            logger.warning("Scene has no scene_id; cannot start")
            return

        with self._lock:
            if self._stopped:
                logger.info("Registry is stopped; not starting scene %s", scene_id)
                return
            if scene_id in self._entries:
                logger.info("Scene %s already running; reloading instead", scene_id)
                reload = True
            elif len(self._entries) >= self._max_scenes:
                logger.warning(
                    "MAX_SCENES (%d) reached; not starting scene %s "
                    "(increase MAX_SCENES or delete a scene)",
                    self._max_scenes, scene_id,
                )
                return
            else:
                reload = False

        if reload:
            self.reload(cfg)
            return

        if not cfg.rtsp_url:
            logger.error(
                "Scene %s has no camera_ip_address; cannot start capture", scene_id
            )
            return
        if cfg.reference_image is None:
            logger.error(
                "Scene %s has no reference image; cannot start pipeline", scene_id
            )
            return

        logger.info(
            "Starting scene %s — version=%s zones=%d classes=%s",
            scene_id, cfg.version, len(cfg.red_zones), cfg.forbidden_classes,
        )
        capture = FrameCapture(cfg.rtsp_url)
        capture.start()
        try:
            pipeline = DetectionPipeline(
                capture=capture,
                api_base=self._api_base,
                initial_config=cfg,
                shared_device=self._shared_device,
            )
            # Subscribe any global handlers (e.g. SampleSaver) before start so
            # no early event is missed.
            with self._lock:
                for handler in self._global_subscribers:
                    pipeline.subscribe(handler)
            pipeline.start()
        except Exception:
            logger.exception("Failed to start pipeline for scene %s; stopping capture", scene_id)
            capture.stop()
            return

        with self._lock:
            if self._stopped:
                # Registry was shut down while we were starting — clean up.
                pass
            self._entries[scene_id] = {
                "capture": capture,
                "pipeline": pipeline,
                "rtsp_url": cfg.rtsp_url,
            }
        if self._stopped:
            pipeline.stop()
            capture.stop()
            with self._lock:
                self._entries.pop(scene_id, None)

    def reload(self, cfg: SceneConfig) -> None:
        scene_id = cfg.scene.get("scene_id")
        if not scene_id:
            return
        with self._lock:
            entry = self._entries.get(scene_id)
        if entry is None:
            # Was removed concurrently; start fresh instead.
            self.start(cfg)
            return

        # If the camera URL changed, the capture can't be hot-swapped — restart
        # the whole scene (capture + pipeline). Otherwise just reload the
        # pipeline's detectors/occlusion (the cheap path).
        if cfg.rtsp_url and cfg.rtsp_url != entry["rtsp_url"]:
            logger.info(
                "Scene %s camera URL changed; restarting capture+pipeline", scene_id
            )
            self._stop_entry(scene_id, entry)
            with self._lock:
                self._entries.pop(scene_id, None)
            self.start(cfg)
            return

        if cfg.reference_image is None:
            logger.warning(
                "Refusing to reload scene %s: new config has no reference image", scene_id
            )
            return

        try:
            entry["pipeline"].reload_config(cfg)
        except Exception:
            logger.exception("Failed to reload scene %s", scene_id)

    def stop(self, scene_id: str) -> None:
        with self._lock:
            entry = self._entries.pop(scene_id, None)
        if entry is None:
            return
        self._stop_entry(scene_id, entry)

    def _stop_entry(self, scene_id: str, entry: dict) -> None:
        logger.info("Stopping scene %s", scene_id)
        try:
            entry["pipeline"].stop()
        except Exception:
            logger.exception("Error stopping pipeline for scene %s", scene_id)
        try:
            entry["capture"].stop()
        except Exception:
            logger.exception("Error stopping capture for scene %s", scene_id)

    def stop_all(self) -> None:
        with self._lock:
            self._stopped = True
            entries = list(self._entries.items())
            self._entries.clear()
        for scene_id, entry in entries:
            self._stop_entry(scene_id, entry)
