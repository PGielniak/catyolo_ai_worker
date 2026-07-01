"""Action handlers for DetectionEvent dispatch.

Each handler type wraps one action_id (one row in the `actions` table) and
processes events on its own background thread, so the pipeline hot loop is
never blocked on network/disk I/O.

Public surface:
    SmbUploaderHandler        - native SMB2/3 upload (smbprotocol)
    TelegramSenderHandler     - Telegram Bot API
    WebhookDispatcherHandler  - generic HTTP POST with 4 auth modes
"""

from detector.handlers.base import BaseActionHandler
from detector.handlers.smb_uploader import SmbUploaderHandler
from detector.handlers.telegram_sender import TelegramSenderHandler
from detector.handlers.webhook_dispatcher import WebhookDispatcherHandler

__all__ = [
    "BaseActionHandler",
    "SmbUploaderHandler",
    "TelegramSenderHandler",
    "WebhookDispatcherHandler",
]
