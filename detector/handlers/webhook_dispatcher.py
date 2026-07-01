"""Webhook alert handler.

POSTs a multipart/form-data payload to a configured URL. Supports four
auth modes (none / basic / API key / OAuth2 client-credentials). The
receiver always gets:
  - metadata  (JSON string in a form field)
  - image     (JPEG file, annotated frame when available)
  - trigger / detected_class / timestamp  (form fields)
"""

import json
import logging
import threading
import time
from typing import Optional, Tuple

import requests

from detector.events import DetectionEvent
from detector.handlers.base import BaseActionHandler

logger = logging.getLogger(__name__)


class WebhookDispatcherHandler(BaseActionHandler):
    MAX_RETRIES = 2  # attempts beyond the first (so up to 3 total POSTs)
    RETRY_BACKOFF = (1.0, 2.0)  # seconds between retries
    REQUEST_TIMEOUT = 15.0

    def __init__(self, action_id: str, action_name: str, config: dict):
        url = (config.get("webhookUrl") or "").strip()
        if not url:
            raise ValueError("WebhookDispatcherHandler requires webhookUrl")
        self._url = url
        self._auth_type = (config.get("webhookAuthType") or "none").strip()
        self._username = config.get("webhookUsername")
        self._password = config.get("webhookPassword")
        self._api_key = config.get("webhookApiKey")
        self._client_id = config.get("webhookClientId")
        self._client_secret = config.get("webhookClientSecret")
        self._token_endpoint = config.get("webhookTokenEndpoint")

        if self._auth_type not in ("none", "basicAuth", "apiKey", "oauth2"):
            raise ValueError(f"Unsupported webhookAuthType: {self._auth_type!r}")

        self._session = requests.Session()
        self._session_lock = threading.Lock()
        # OAuth2 token cache
        self._oauth_token: Optional[str] = None
        self._oauth_token_expiry: float = 0.0
        self._oauth_lock = threading.Lock()

        super().__init__(action_id, action_name)
        logger.info(
            "WebhookDispatcherHandler configured — url=%s auth=%s",
            self._url, self._auth_type,
        )

    # ------------------------------------------------------------------ #

    def _thread_name(self) -> str:
        return f"webhook-{self._action_id[:8]}"

    def _deliver(self, event: DetectionEvent, payload: dict) -> None:
        data, headers, files = self._build_request(event, payload)
        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES + 1):
            with self._session_lock:
                try:
                    r = self._session.post(
                        self._url, data=data, files=files, headers=headers, timeout=self.REQUEST_TIMEOUT,
                    )
                except Exception as e:
                    last_exc = e
                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.RETRY_BACKOFF[attempt])
                        continue
                    raise
            if 200 <= r.status_code < 300:
                logger.info(
                    "Webhook send ok — action=%s status=%d trigger=%s",
                    self._action_id, r.status_code, event.trigger,
                )
                return
            # Non-2xx; log and retry
            logger.warning(
                "Webhook send non-2xx — action=%s status=%d body=%s",
                self._action_id, r.status_code, r.text[:200],
            )
            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_BACKOFF[attempt])
                continue
            raise RuntimeError(
                f"Webhook {self._url} returned {r.status_code} after {self.MAX_RETRIES + 1} attempts"
            )
        # Unreachable, but keep mypy/linter happy.
        if last_exc:
            raise last_exc

    # ------------------------------------------------------------------ #
    # Request construction
    # ------------------------------------------------------------------ #

    def _build_request(
        self, event: DetectionEvent, payload: dict
    ) -> Tuple[dict, dict, dict]:
        files: dict = {}
        image = payload["annotated_jpeg"] or payload["raw_jpeg"]
        if image:
            files["image"] = ("event.jpg", image, "image/jpeg")

        data = {
            "metadata": json.dumps(payload["metadata"], ensure_ascii=False),
            "trigger": event.trigger,
            "scene_id": event.scene_id or "",
            "detected_class": event.detected_class or "",
            "timestamp": event.timestamp.isoformat(),
        }

        headers: dict = {}

        if self._auth_type == "none":
            pass
        elif self._auth_type == "basicAuth":
            import base64
            token = base64.b64encode(
                f"{self._username or ''}:{self._password or ''}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        elif self._auth_type == "apiKey":
            headers["Authorization"] = f"Bearer {self._api_key or ''}"
        elif self._auth_type == "oauth2":
            token = self._get_oauth_token()
            headers["Authorization"] = f"Bearer {token}"

        return data, headers, files

    # ------------------------------------------------------------------ #
    # OAuth2 client_credentials
    # ------------------------------------------------------------------ #

    def _get_oauth_token(self) -> str:
        with self._oauth_lock:
            now = time.monotonic()
            if self._oauth_token and now < self._oauth_token_expiry - 30:
                return self._oauth_token
            if not self._token_endpoint or not self._client_id or not self._client_secret:
                raise RuntimeError("OAuth2 config missing token_endpoint/client_id/client_secret")
            r = self._session.post(
                self._token_endpoint,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                timeout=self.REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            tok = r.json()
            access = tok.get("access_token")
            if not access:
                raise RuntimeError(f"OAuth2 token response missing access_token: {tok}")
            self._oauth_token = access
            self._oauth_token_expiry = now + float(tok.get("expires_in") or 3600)
            return access

    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        super().stop()
        try:
            self._session.close()
        except Exception:
            pass
