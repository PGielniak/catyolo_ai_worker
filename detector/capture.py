import threading
import time
import cv2


class FrameCapture:
    def __init__(self, rtsp_url):
        self._url = rtsp_url
        self._cap = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
    
    def start(self):
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self):
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
    
    def get(self):
        with self._lock:
            return self._latest.copy() if self._latest is not None else None
