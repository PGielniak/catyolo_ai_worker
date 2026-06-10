import logging
import os
import signal
import threading

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
from detector.capture import FrameCapture
from detector.config import SceneConfig
from detector.config_watcher import ConfigWatcher
from detector.events import set_dispatcher
from detector.handlers.registry import ActionHandlerRegistry
from detector.handlers.sample_saver import SampleSaverHandler
from detector.pipeline import DetectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("catyolo_worker")


def _fetch_initial_config(api_base: str) -> SceneConfig:
    """Synchronously fetch the first scene from the backend and build the
    initial SceneConfig. Raises if nothing is available — we can't run
    without a scene."""
    import requests

    r = requests.get(f"{api_base.rstrip('/')}/scene", timeout=5.0)
    r.raise_for_status()
    scenes = r.json()
    if not scenes:
        raise RuntimeError(
            f"Backend at {api_base} returned no scenes. Create a scene in the UI first."
        )
    logger.info(
        "Loaded initial scene — scene_id=%s, version=%s, zones=%d",
        scenes[0].get("scene_id"),
        scenes[0].get("version"),
        len(scenes[0].get("red_zones") or []),
    )
    return SceneConfig.from_scene_dict(scenes[0])


def main():
    capture = FrameCapture(os.getenv("RTSP_URL", ""))
    capture.start()
    logger.info("frame capture started")

    api_base = os.getenv("API_BASE", "http://localhost:8100")
    initial_config = _fetch_initial_config(api_base)

    pipeline = DetectionPipeline(
        capture=capture,
        api_base=api_base,
        initial_config=initial_config,
    )
    pipeline.start()
    logger.info("detection pipeline started")

    if os.getenv("ENABLE_SAMPLE_SAVER", "false").lower() == "true":
        samples_dir = os.getenv("SAMPLES_DIR", "./samples")
        pipeline.subscribe(SampleSaverHandler(samples_dir))
        logger.info("sample saver enabled - saving to %s", samples_dir)

    if os.getenv("ENABLE_STREAM", "false").lower() == "true":
        from detector.stream import run_stream
        port = int(os.getenv("STREAM_PORT", "5001"))
        threading.Thread(target=run_stream, args=(pipeline, port), daemon=True).start()
        logger.info("debug stream started on port %s", port)

    # ── Action dispatch ─────────────────────────────────────────────────
    # The registry owns one handler per known action_id. The ActionsWatcher
    # polls /action/ every 2s, diffs the snapshot, and pushes updates to
    # the registry. The registry is registered as the pipeline-level
    # dispatcher; the pipeline calls it with (event, action_ids) for every
    # DetectionEvent.
    action_registry = ActionHandlerRegistry()
    set_dispatcher(action_registry.dispatch_event)

    actions_watcher = ActionsWatcher(
        api_base=api_base,
        on_change=action_registry.set_actions,
    )
    actions_watcher.start()
    logger.info("actions watcher started")

    config_watcher = ConfigWatcher(
        api_base=api_base,
        on_change=pipeline.reload_config,
    )
    config_watcher.start()
    logger.info("config watcher started")

    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    try:
        shutdown.wait()
    finally:
        config_watcher.stop()
        actions_watcher.stop()
        action_registry.stop()
        pipeline.stop()
        capture.stop()
        logger.info("shutdown complete")


if __name__ == "__main__":
    main()

