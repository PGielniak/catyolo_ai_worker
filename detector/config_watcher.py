import logging
import os
import threading
from typing import Callable, Optional

import requests

from detector.config import SceneConfig

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """Background thread that polls the backend for scene-config changes and
    calls `on_change(new_config)` whenever a new version is observed.

    Mechanism:
      - GET {api_base}/scene/version  -> small integer
      - if it moves, GET {api_base}/scene/  -> first scene -> build SceneConfig
      - call on_change(config)

    First tick after `start()` always fires (so the watcher can also be used as
    a "first load" path). The pipeline uses that to converge with the same code
    path that handles live reloads.
    """

    def __init__(
        self,
        api_base: str,
        on_change: Callable[[SceneConfig], None],
        poll_interval: Optional[float] = None,
    ):
        self._api_base = api_base.rstrip("/")
        self._on_change = on_change
        self._poll_interval = (
            poll_interval
            if poll_interval is not None
            else float(os.getenv("CONFIG_POLL_INTERVAL", "2.0"))
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_version: Optional[int] = None
        # When True, the next poll fires on_change regardless of version. Set
        # to True on start() so we always emit at least once.
        self._force_emit = True
        self._session = requests.Session()

    def start(self):
        if self._thread is not None:
            logger.warning("ConfigWatcher already started")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="config-watcher"
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        try:
            self._session.close()
        except Exception:
            pass

    def _run(self):
        logger.info(
            "ConfigWatcher started — polling %s/scene/version every %.1fs",
            self._api_base,
            self._poll_interval,
        )
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("ConfigWatcher tick failed")
            # Interruptible sleep
            self._stop_event.wait(self._poll_interval)

    def _tick(self):
        version = self._fetch_version()
        if version is None:
            return

        if not self._force_emit and version == self._last_version:
            return

        scene = self._fetch_first_scene()
        if scene is None:
            logger.debug("No scenes available; skipping reload")
            return

        try:
            config = SceneConfig.from_scene_dict(scene)
        except Exception:
            logger.exception("Failed to build SceneConfig from scene dict")
            return

        previous = self._last_version
        self._last_version = version
        self._force_emit = False

        logger.info(
            "Scene config change detected — version %s -> %s, scene_id=%s, zones=%d",
            previous,
            version,
            config.scene.get("scene_id"),
            len(config.red_zones),
        )
        try:
            self._on_change(config)
        except Exception:
            logger.exception("on_change callback raised; will retry next tick")

    def _fetch_version(self) -> Optional[int]:
        try:
            r = self._session.get(f"{self._api_base}/scene/version", timeout=2.0)
            r.raise_for_status()
            data = r.json()
            v = data.get("version")
            return int(v) if v is not None else None
        except Exception as e:
            logger.debug("Failed to fetch /scene/version: %s", e)
            return None

    def _fetch_first_scene(self) -> Optional[dict]:
        try:
            r = self._session.get(f"{self._api_base}/scene", timeout=5.0)
            r.raise_for_status()
            scenes = r.json()
        except Exception:
            logger.exception("Failed to fetch /scene")
            return None
        if not scenes:
            return None
        return scenes[0]
