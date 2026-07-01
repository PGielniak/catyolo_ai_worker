"""Registry of action handlers.

The ActionsWatcher polls the backend for the full action set, hashes each
action's (type, config) to detect changes, and calls
`registry.set_actions(actions)` with the new snapshot. The registry diffs
against the previous set and starts/stops the right handler instances.

The pipeline calls `registry.dispatch_event(event, action_ids)` for every
detection event (zone-level and global-prompt). The registry looks up the
matching handlers and fans out.
"""

import hashlib
import json
import logging
from typing import Optional

from detector.events import DetectionEvent, set_dispatcher
from detector.handlers import (
    SmbUploaderHandler,
    TelegramSenderHandler,
    WebhookDispatcherHandler,
)

logger = logging.getLogger(__name__)


# Action types we know how to dispatch. Unknown types are silently ignored
# (the registry stores them by ID but never starts a handler for them).
_HANDLER_FACTORIES = {
    "telegram": lambda action_id, name, cfg: TelegramSenderHandler(action_id, name, cfg),
    "webhook": lambda action_id, name, cfg: WebhookDispatcherHandler(action_id, name, cfg),
    "smbFileshare": lambda action_id, name, cfg: SmbUploaderHandler(action_id, name, cfg),
}


def _config_hash(action_type: str, action_config: dict) -> str:
    """Stable hash of (type, config). Used to detect 'this action's settings
    changed, restart the handler'."""
    payload = json.dumps(
        {"t": action_type, "c": action_config},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ActionHandlerRegistry:
    """Owns one handler instance per known action_id. Lifecycle is driven
    exclusively via set_actions() — call it with the latest snapshot from
    the backend on every poll where the set has changed.
    """

    def __init__(self):
        # id -> handler instance
        self._handlers: dict[str, object] = {}
        # id -> (type, config_hash) so we can detect same-id-different-config
        self._signatures: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def set_actions(self, actions: list[dict]) -> None:
        """Reconcile the running handler set with the backend's current
        action list. Stops handlers for removed actions, restarts those
        whose config changed, starts new ones, and silently drops unknown
        action_type values.
        """
        actions = actions or []
        new_ids = {a["action_id"] for a in actions}
        new_signatures = {
            a["action_id"]: (a.get("action_type") or "", _config_hash(a.get("action_type") or "", a.get("action_config") or {}))
            for a in actions
        }

        # 1. Stop handlers for removed IDs
        for removed_id in self._handlers.keys() - new_ids:
            self._stop_handler(removed_id)

        # 2. (Re)start handlers that are new OR whose config hash changed
        for a in actions:
            action_id = a["action_id"]
            action_type = a.get("action_type") or ""
            sig = new_signatures[action_id]
            if sig != self._signatures.get(action_id):
                if action_id in self._handlers:
                    self._stop_handler(action_id)
                if action_type in _HANDLER_FACTORIES:
                    self._start_handler(action_id, a.get("action_name") or action_id, action_type, a.get("action_config") or {})
                else:
                    logger.info(
                        "Ignoring action %s of unknown type %r (no handler)",
                        action_id, action_type,
                    )

    def stop(self) -> None:
        for action_id in list(self._handlers.keys()):
            self._stop_handler(action_id)
        set_dispatcher(None)

    # ------------------------------------------------------------------ #
    # Dispatcher entry point
    # ------------------------------------------------------------------ #

    def dispatch_event(self, event: DetectionEvent, action_ids: list[str]) -> None:
        """Called by the pipeline for every detection event with the list
        of action IDs that should fire. No-op if action_ids is empty.
        """
        if not action_ids:
            return
        for action_id in action_ids:
            handler = self._handlers.get(action_id)
            if handler is None:
                # Action was deleted in the UI, or its type isn't supported.
                continue
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler dispatch raised — action=%s trigger=%s",
                    action_id, event.trigger,
                )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _start_handler(self, action_id: str, action_name: str, action_type: str, action_config: dict) -> None:
        factory = _HANDLER_FACTORIES[action_type]
        try:
            handler = factory(action_id, action_name, action_config)
        except Exception:
            logger.exception(
                "Failed to start %s handler for action %s (%s) — will retry on next set_actions()",
                action_type, action_id, action_name,
            )
            return
        self._handlers[action_id] = handler
        self._signatures[action_id] = (action_type, _config_hash(action_type, action_config))

    def _stop_handler(self, action_id: str) -> None:
        handler = self._handlers.pop(action_id, None)
        self._signatures.pop(action_id, None)
        if handler is None:
            return
        try:
            handler.stop()
        except Exception:
            logger.exception("Error stopping handler for action %s", action_id)
