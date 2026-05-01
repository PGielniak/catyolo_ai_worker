import time
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse


def run_stream(pipeline, port: int = 5001):
    app = FastAPI()

    def _frames():
        while True:
            frame = pipeline.get_annotated()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(1 / 15)

    @app.get("/feed")
    def feed():
        return StreamingResponse(_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return "<html><body><img src='/feed' style='max-width:100%'></body></html>"

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
