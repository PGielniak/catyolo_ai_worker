import os
import threading
import time
import cv2
import ctypes

# Use TCP for RTSP transport (mirrors routes/frame.py in the backend). TCP
# avoids the packet loss / tearing common with UDP MJPEG streams. Set once
# at import so every cv2.VideoCapture in this process uses it.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

def set_thread_name(name: str):
    """Set the OS-visible thread name. Linux only, max 15 chars."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)  # 15 = PR_SET_NAME
    except Exception:
        pass

class FrameCapture:
    def __init__(self, rtsp_url):
        self._url = rtsp_url
        self._cap = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last_frame_time: float = 0.0

    def start(self):
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def last_frame_age(self) -> float | None:
        """Seconds since the last successfully decoded frame, or None if no frame yet."""
        t = self._last_frame_time
        return (time.monotonic() - t) if t > 0.0 else None

    def _run(self):
        set_thread_name("capture")
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                # reconnect logic
                time.sleep(0.5)
                self._cap.release()
                self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            with self._lock:
                self._latest = frame
                self._last_frame_time = time.monotonic()

    def get(self):
        with self._lock:
            return self._latest.copy() if self._latest is not None else None
