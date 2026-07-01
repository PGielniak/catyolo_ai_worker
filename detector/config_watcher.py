"""ConfigWatcher — polls the backend for per-scene config changes (WS2).

Replaces the legacy single-global-version poll with a per-scene version diff:
  - GET {api_base}/scene/version -> {"version": <max>,
                                     "scenes": [{"scene_id", "version"}]}
  - compare per-scene versions against the last known set
  - for changed/new scenes, fetch full scene dicts via /scene/internal/
    (carries camera_password, needed to build the RTSP URL)
  - call on_change(changed: list[SceneConfig], removed: list[str])

First tick after start() always fires all current scenes as "changed" (the
same force-emit pattern as ActionsWatcher and the legacy ConfigWatcher), so
the watcher also serves as the initial load path — main() doesn't need to
start scenes itself.

Reuses the diff-and-reconcile shape proven in ActionsWatcher: poll full list,
compute per-item delta, only invoke the callback when something moved, and
swallow per-tick exceptions so the poll loop survives a bad tick.
"""

import logging
import os
import threading
from typing import Callable, Optional

import requests

from detector.config import SceneConfig

logger = logging.getLogger(__name__)


class ConfigWatcher:
    def __init__(
        self,
        api_base: str,
        on_change: Callable[[list[SceneConfig], list[str]], None],
        poll_interval: Optional[float] = None,
        api_key: Optional[str] = None,
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
        self._last_versions: dict[str, int] = {}
        self._force_emit = True
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key

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
            self._stop_event.wait(self._poll_interval)

    def _tick(self):
        versions = self._fetch_versions()
        if versions is None:
            return

        current_ids = set(versions.keys())
        last_ids = set(self._last_versions.keys())

        # Quick path: nothing changed since last tick (and not the first tick).
        if not self._force_emit and versions == self._last_versions:
            return

        changed_ids = [
            sid for sid, v in versions.items()
            if self._force_emit or self._last_versions.get(sid) != v
        ]
        removed_ids = list(last_ids - current_ids)

        if not changed_ids and not removed_ids:
            # versions dict identity already checked above, so this only happens
            # on the first tick with an empty scene set.
            self._last_versions = versions
            self._force_emit = False
            return

        scenes_by_id = self._fetch_scenes(changed_ids) if changed_ids else {}
        changed: list[SceneConfig] = []
        for sid in changed_ids:
            scene = scenes_by_id.get(sid)
            if scene is None:
                logger.debug("Scene %s vanished between version and detail fetch; skipping", sid)
                continue
            try:
                changed.append(SceneConfig.from_scene_dict(scene))
            except Exception:
                logger.exception("Failed to build SceneConfig for scene %s", sid)

        previous = self._last_versions
        self._last_versions = versions
        self._force_emit = False

        logger.info(
            "Scene config change — changed=%d removed=%d (previous=%s)",
            len(changed), len(removed_ids),
            {k: v for k, v in previous.items()} or "—",
        )
        try:
            self._on_change(changed, removed_ids)
        except Exception:
            logger.exception("on_change callback raised; will retry next tick")

    def _fetch_versions(self) -> Optional[dict[str, int]]:
        try:
            r = self._session.get(f"{self._api_base}/scene/version", timeout=2.0)
            r.raise_for_status()
            data = r.json()
            scenes = data.get("scenes") or []
            return {s["scene_id"]: int(s.get("version") or 0) for s in scenes}
        except Exception as e:
            logger.debug("Failed to fetch /scene/version: %s", e)
            return None

    def _fetch_scenes(self, scene_ids: list[str]) -> dict[str, dict]:
        # The backend has no per-scene internal endpoint; /scene/internal/
        # returns all scenes with credentials. Fetch once and filter locally —
        # the payload is small (≤ MAX_SCENES scenes).
        try:
            r = self._session.get(f"{self._api_base}/scene/internal/", timeout=5.0)
            r.raise_for_status()
            scenes = r.json()
        except Exception:
            logger.exception("Failed to fetch /scene/internal/")
            return {}
        wanted = set(scene_ids)
        return {
            s.get("scene_id"): s
            for s in scenes
            if s.get("scene_id") in wanted
        }
