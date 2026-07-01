import json
import logging
import queue
import threading
from pathlib import Path

import cv2

from detector.events import DetectionEvent

logger = logging.getLogger(__name__)


class SampleSaverHandler:
    """Saves every DetectionEvent to a timestamped folder under samples_dir.

    Each event produces a folder named  YYYYMMDD_HHMMSS_ffffff_<trigger>/
    containing:
      raw_frame.jpg   - the undecorated frame (when present)
      annotated.jpg   - frame with YOLO/zone overlays
      metadata.json   - all non-image fields

    Disk I/O runs on a background thread so the pipeline is never blocked.
    """

    def __init__(self, samples_dir):
        self._samples_dir = Path(samples_dir)
        self._samples_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="sample-saver")
        self._thread.start()
        logger.info("SampleSaverHandler started - saving to %s", self._samples_dir)

    def __call__(self, event):
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("SampleSaverHandler queue full - dropping event trigger=%s", event.trigger)

    def _worker(self):
        while True:
            event = self._queue.get()
            try:
                self._save(event)
            except Exception:
                logger.exception("SampleSaverHandler failed to save event trigger=%s", event.trigger)

    def _save(self, event):
        ts = event.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        # Nest under scene_id so concurrent scenes never collide on
        # timestamp+trigger folder names, and samples are browsable per camera.
        scene_dir = self._samples_dir / (event.scene_id or "unknown_scene")
        scene_dir.mkdir(parents=True, exist_ok=True)
        folder = scene_dir / f"{ts}_{event.trigger}"
        folder.mkdir(parents=True, exist_ok=True)

        if event.raw_frame is not None:
            cv2.imwrite(str(folder / "raw_frame.jpg"), event.raw_frame)

        if event.annotated_image is not None:
            cv2.imwrite(str(folder / "annotated.jpg"), event.annotated_image)

        meta = {
            "trigger": event.trigger,
            "timestamp": event.timestamp.isoformat(),
            "scene_id": event.scene_id,
            "detected_class": event.detected_class,
            "vlm_prompt": event.vlm_prompt,
            "vlm_answer": event.vlm_answer,
            "zone": {
                k: v for k, v in event.zone.items()
                if not isinstance(v, (bytes, bytearray))
            } if event.zone else None,
        }
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.debug("Saved sample %s/%s", scene_dir.name, folder.name)
