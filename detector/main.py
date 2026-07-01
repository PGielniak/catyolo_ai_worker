import logging
import os
import signal
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Load .env before hailo_platform is imported so HAILORT_LOGGER_PATH takes effect.
load_dotenv()
import os as _os
_os.makedirs(_os.path.expanduser("~/.local/share/catyolo/logs"), exist_ok=True)
_os.environ.setdefault(
    "HAILORT_LOGGER_PATH",
    _os.path.expanduser("~/.local/share/catyolo/logs/hailort.log"),
)
del _os

from detector.actions_watcher import ActionsWatcher
from detector.config_watcher import ConfigWatcher
from detector.events import set_dispatcher
from detector.handlers.registry import ActionHandlerRegistry
from detector.handlers.sample_saver import SampleSaverHandler
from detector.inference.factory import create_shared_device, release_shared_device
from detector.scene_registry import ScenePipelineRegistry

# ── Logging ────────────────────────────────────────────────────────────────
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("catyolo_worker")

log_file = os.getenv("LOG_FILE_PATH")
if log_file:
    log_dir = Path(log_file).parent
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)


def main():
    api_base = os.getenv("API_BASE", "http://localhost:8100")
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        logger.warning("API_KEY not set — requests to the backend will be unauthenticated")

    max_scenes = int(os.getenv("MAX_SCENES", "3"))

    # Shared Hailo VDevice — all per-scene backends multiplex over this one
    # device via the HailoRT ROUND_ROBIN scheduler (group_id="SHARED"). If
    # creation fails, fall back to per-backend device ownership (legacy path);
    # that preserves the previous behaviour on setups where a shared VDevice
    # can't be opened.
    try:
        shared_device = create_shared_device()
    except Exception:
        logger.exception(
            "Failed to create shared VDevice; falling back to per-backend "
            "device ownership (multi-camera may not be possible)"
        )
        shared_device = None

    registry = ScenePipelineRegistry(
        api_base=api_base,
        shared_device=shared_device,
        max_scenes=max_scenes,
    )

    # Optional sample saver — one instance subscribed to ALL pipelines
    # (including scenes added later via hot-reload).
    if os.getenv("ENABLE_SAMPLE_SAVER", "false").lower() == "true":
        samples_dir = os.getenv("SAMPLES_DIR", "./samples")
        registry.subscribe(SampleSaverHandler(samples_dir))
        logger.info("sample saver enabled - saving to %s", samples_dir)

    # Optional debug stream — one port serving all scenes (/feed/{scene_id}).
    if os.getenv("ENABLE_STREAM", "false").lower() == "true":
        from detector.stream import run_stream
        port = int(os.getenv("STREAM_PORT", "5001"))
        threading.Thread(target=run_stream, args=(registry, port), daemon=True).start()
        logger.info("debug stream started on port %s", port)

    # Shared action handler registry — scene-agnostic (keyed by action_id, so
    # one registry serves events from every scene). The dispatcher is the
    # process-wide singleton the pipeline calls after emitting on its bus.
    action_registry = ActionHandlerRegistry()
    set_dispatcher(action_registry.dispatch_event)

    actions_watcher = ActionsWatcher(
        api_base=api_base,
        on_change=action_registry.set_actions,
        api_key=api_key,
    )
    actions_watcher.start()
    logger.info("actions watcher started")

    # ConfigWatcher does per-scene version diff and calls registry.apply(
    # changed, removed). Its first tick force-emits all current scenes, so it
    # also serves as the initial load path (main() doesn't start scenes itself).
    config_watcher = ConfigWatcher(
        api_base=api_base,
        on_change=registry.apply,
        api_key=api_key,
    )
    config_watcher.start()
    logger.info("config watcher started (max_scenes=%d)", max_scenes)

    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    try:
        shutdown.wait()
    finally:
        # Stop in dependency order: watchers -> action registry -> per-scene
        # pipelines+captures -> shared VDevice.
        config_watcher.stop()
        actions_watcher.stop()
        action_registry.stop()
        registry.stop_all()
        release_shared_device(shared_device)
        logger.info("shutdown complete")


if __name__ == "__main__":
    main()
