"""ActionsWatcher — polls the backend for the action set and reconciles the
handler registry whenever it changes.

Mirrors ConfigWatcher: a background thread, an interruptible sleep, and a
GET on the configured interval. We don't have a /action/version endpoint,
so we hash the full payload and only call back when the hash moves.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Callable, Optional

import requests


logger = logging.getLogger(__name__)


def _snapshot_hash(actions: list[dict]) -> str:
    norm = sorted(
        (
            a.get("action_id"),
            a.get("action_type"),
            json.dumps(a.get("action_config") or {}, sort_keys=True, ensure_ascii=False, default=str),
        )
        for a in actions
    )
    payload = json.dumps(norm, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ActionsWatcher:
    """Background thread that polls `GET {api_base}/action/internal/` and calls
    `on_change(actions)` whenever the snapshot changes.

    Uses the internal endpoint to receive full action credentials.
    First tick after `start()` always fires (same pattern as ConfigWatcher).
    """

    def __init__(
        self,
        api_base: str,
        on_change: Callable[[list[dict]], None],
        poll_interval: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        self._api_base = api_base.rstrip("/")
        self._on_change = on_change
        self._poll_interval = (
            poll_interval
            if poll_interval is not None
            else float(os.getenv("ACTIONS_POLL_INTERVAL", "2.0"))
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_hash: Optional[str] = None
        self._force_emit = True
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    def start(self) -> None:
        if self._thread is not None:
            logger.warning("ActionsWatcher already started")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="actions-watcher",
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        try:
            self._session.close()
        except Exception:
            pass

    def _run(self) -> None:
        logger.info(
            "ActionsWatcher started — polling %s/action/internal/ every %.1fs",
            self._api_base, self._poll_interval,
        )
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("ActionsWatcher tick failed")
            self._stop_event.wait(self._poll_interval)

    def _tick(self) -> None:
        actions = self._fetch_actions()
        if actions is None:
            return
        h = _snapshot_hash(actions)
        if not self._force_emit and h == self._last_hash:
            return
        previous = self._last_hash
        self._last_hash = h
        self._force_emit = False
        logger.info(
            "Action set changed — hash %s -> %s, count=%d",
            (previous or "—")[:8], h[:8], len(actions),
        )
        try:
            self._on_change(actions)
        except Exception:
            logger.exception("ActionsWatcher on_change callback raised; will retry next tick")

    def _fetch_actions(self) -> Optional[list[dict]]:
        try:
            r = self._session.get(f"{self._api_base}/action/internal/", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                logger.warning("Unexpected /action/internal/ payload (not a list): %r", type(data))
                return None
            return data
        except Exception as e:
            logger.debug("Failed to fetch /action/internal/: %s", e)
            return None
