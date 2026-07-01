"""Telegram bot alert handler.

Sends a photo + caption to a configured chat via the Bot API. Uses
annotated.jpg when available so the user immediately sees the bounding
boxes; falls back to raw_frame.jpg; falls back to a text message if no
image is on the event.
"""

import logging
import threading
from typing import Optional

import requests

from detector.events import DetectionEvent
from detector.handlers.base import BaseActionHandler

logger = logging.getLogger(__name__)


DEFAULT_TEMPLATE = "🚨 {scene}: {trigger} detected: {class} at {ts}"


def _render_template(template: str, event: DetectionEvent) -> str:
    return template.format(
        trigger=event.trigger,
        class_=event.detected_class or "object",
        cls=event.detected_class or "object",
        scene=event.scene_id or "unknown-scene",
        ts=event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    )


class TelegramSenderHandler(BaseActionHandler):
    API_BASE = "https://api.telegram.org"

    def __init__(self, action_id: str, action_name: str, config: dict):
        token = (config.get("telegramBotToken") or "").strip()
        chat_id = (config.get("telegramChatId") or "").strip()
        if not token or not chat_id:
            raise ValueError(
                "TelegramSenderHandler requires telegramBotToken and telegramChatId"
            )
        self._token = token
        self._chat_id = chat_id
        self._template = (config.get("telegramMessageTemplate") or "").strip() or DEFAULT_TEMPLATE
        self._session = requests.Session()
        self._session_lock = threading.Lock()
        super().__init__(action_id, action_name)
        # Avoid ever logging the token.
        logger.info(
            "TelegramSenderHandler configured — chat=%s template=%r",
            self._chat_id, self._template,
        )

    def _thread_name(self) -> str:
        return f"telegram-{self._action_id[:8]}"

    def _deliver(self, event: DetectionEvent, payload: dict) -> None:
        caption = _render_template(self._template, event)
        # Prefer annotated (with bbox overlays) for visual confirmation.
        image_bytes: Optional[bytes] = (
            payload["annotated_jpeg"]
            or payload["raw_jpeg"]
            or None
        )
        with self._session_lock:
            try:
                if image_bytes:
                    self._send_photo(image_bytes, caption)
                else:
                    self._send_message(caption)
                logger.info(
                    "Telegram send ok — action=%s chat=%s trigger=%s",
                    self._action_id, self._chat_id, event.trigger,
                )
            except Exception:
                # Reset the session so the next event re-handshakes.
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = requests.Session()
                raise

    def _send_photo(self, image_bytes: bytes, caption: str) -> None:
        url = f"{self.API_BASE}/bot{self._token}/sendPhoto"
        files = {"photo": ("event.jpg", image_bytes, "image/jpeg")}
        data = {"chat_id": self._chat_id, "caption": caption[:1024]}
        r = self._session.post(url, data=data, files=files, timeout=15)
        if not r.ok:
            raise RuntimeError(f"Telegram sendPhoto failed: {r.status_code} {r.text[:200]}")

    def _send_message(self, text: str) -> None:
        url = f"{self.API_BASE}/bot{self._token}/sendMessage"
        data = {"chat_id": self._chat_id, "text": text[:4096]}
        r = self._session.post(url, data=data, timeout=15)
        if not r.ok:
            raise RuntimeError(f"Telegram sendMessage failed: {r.status_code} {r.text[:200]}")

    def stop(self) -> None:
        super().stop()
        try:
            self._session.close()
        except Exception:
            pass
