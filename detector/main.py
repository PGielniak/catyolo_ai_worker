import logging
import os
import signal
import threading

from dotenv import load_dotenv

from detector.capture import FrameCapture
from detector.pipeline import DetectionPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("catyolo_worker")


def main():
    capture = FrameCapture(os.getenv("RTSP_URL", ""))
    capture.start()
    logger.info("frame capture started")

    pipeline = DetectionPipeline(
        capture=capture,
        api_base=os.getenv("API_BASE", "http://localhost:8100"),
    )
    pipeline.start()
    logger.info("detection pipeline started")

    if os.getenv("ENABLE_STREAM", "false").lower() == "true":
        from detector.stream import run_stream
        port = int(os.getenv("STREAM_PORT", "5001"))
        threading.Thread(target=run_stream, args=(pipeline, port), daemon=True).start()
        logger.info("debug stream started on port %s", port)

    signal.pause()


if __name__ == "__main__":
    main()
