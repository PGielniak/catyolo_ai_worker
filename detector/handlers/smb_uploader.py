"""SMB fileshare upload handler (smbprotocol — SMB2/SMB3).

Writes per-event folders to a remote SMB share. Each event lands in a
timestamped subfolder containing raw_frame.jpg, annotated.jpg, and
metadata.json.

Uses smbprotocol instead of pysmb so the worker is compatible with modern
Samba servers that have SMB1 disabled (the default since Samba 4.11).
"""

import json
import logging

import smbclient
import smbclient.path

from detector.events import DetectionEvent
from detector.handlers.base import BaseActionHandler

logger = logging.getLogger(__name__)

# smbprotocol is verbose at DEBUG; keep only warnings unless the user opts in.
logging.getLogger("smbprotocol").setLevel(logging.WARNING)
logging.getLogger("spnego").setLevel(logging.WARNING)


class SmbUploaderHandler(BaseActionHandler):
    REQUIRED_KEYS = ("smbHost", "smbShare", "smbFolder", "smbUsername", "smbPassword")

    def __init__(self, action_id: str, action_name: str, config: dict):
        missing = [k for k in self.REQUIRED_KEYS if not config.get(k)]
        if missing:
            raise ValueError(
                f"SmbUploaderHandler missing required config keys: {missing}"
            )
        self._host = config["smbHost"]
        self._port = int(config.get("smbPort") or 445)
        self._share = config["smbShare"]
        raw_folder = (config.get("smbFolder") or "").strip().replace("\\", "/")
        self._folder = raw_folder.strip("/")
        self._password = config["smbPassword"]

        # smbprotocol accepts "DOMAIN\username" format for domain auth.
        username = config["smbUsername"]
        domain = (config.get("smbDomain") or "").strip()
        if domain and domain.upper() != "WORKGROUP":
            self._username = f"{domain}\\{username}"
        else:
            self._username = username

        super().__init__(action_id, action_name)
        logger.info(
            "SmbUploaderHandler configured — host=%s:%d share=%s folder=%s user=%s",
            self._host, self._port, self._share, self._folder, config["smbUsername"],
        )

    # ------------------------------------------------------------------ #
    # BaseActionHandler
    # ------------------------------------------------------------------ #

    def _thread_name(self) -> str:
        return f"smb-{self._action_id[:8]}"

    def _deliver(self, event: DetectionEvent, payload: dict) -> None:
        ts = event.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        scene_id = event.scene_id or "unknown_scene"
        subfolder = f"{scene_id}/{ts}_{event.trigger}"
        try:
            self._ensure_session()
            self._upload_files(subfolder, payload)
            logger.info(
                "SMB upload ok — action=%s remote=%s/%s/%s",
                self._action_id, self._share, self._folder, subfolder,
            )
        except Exception:
            # Drop the cached connection so the next event re-handshakes.
            try:
                smbclient.reset_connection_cache()
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------ #
    # SMB plumbing
    # ------------------------------------------------------------------ #

    def _unc(self, *parts: str) -> str:
        r"""Build a UNC path: \\host\share[\part1\part2...]"""
        segments = [self._host, self._share] + [
            p.strip("/\\") for p in parts if p
        ]
        return "\\\\" + "\\".join(segments)

    def _ensure_session(self) -> None:
        smbclient.register_session(
            self._host,
            username=self._username,
            password=self._password,
            port=self._port,
            auth_protocol="negotiate",
        )

    def _upload_files(self, subfolder: str, payload: dict) -> None:
        target = f"{self._folder}/{subfolder}" if self._folder else subfolder

        smbclient.makedirs(self._unc(target), exist_ok=True)

        for filename, data in (
            ("raw_frame.jpg", payload["raw_jpeg"]),
            ("annotated.jpg", payload["annotated_jpeg"]),
        ):
            if not data:
                continue
            with smbclient.open_file(self._unc(target, filename), mode="wb") as fh:
                fh.write(data)

        meta_bytes = json.dumps(
            payload["metadata"], indent=2, ensure_ascii=False
        ).encode("utf-8")
        with smbclient.open_file(self._unc(target, "metadata.json"), mode="wb") as fh:
            fh.write(meta_bytes)

    def stop(self) -> None:
        super().stop()
        try:
            smbclient.reset_connection_cache()
        except Exception:
            pass
