import time
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse


def run_stream(pipeline, port: int = 5001):
    app = FastAPI()

    def _frames():
        while True:
            annotated = pipeline.get_annotated()

            if annotated is None:
                time.sleep(0.05)
                continue

            if pipeline.get_depth_show():
                depth = pipeline.get_depth_viz()
                if depth is not None:
                    dh, dw = depth.shape[:2]
                    ah, aw = annotated.shape[:2]
                    scale = ah / dh
                    depth_resized = cv2.resize(depth, (int(dw * scale), ah), interpolation=cv2.INTER_LINEAR)
                    frame = np.hstack((annotated, depth_resized))
                    cv2.putText(frame, "Annotated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Depth", (aw + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    frame = annotated
            else:
                frame = annotated

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(1 / 15)

    @app.get("/toggle_depth")
    def toggle_depth():
        current = pipeline.get_depth_show()
        pipeline.set_depth_show(not current)
        return {"depth_show": not current}

    @app.get("/feed")
    def feed():
        return StreamingResponse(_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """<html><body style="margin:0;background:#111;color:#fff;font-family:monospace">
<button onclick="fetch('/toggle_depth').then(r=>r.json()).then(d=>this.textContent='Depth: '+(d.depth_show?'ON':'OFF'))" style="margin:10px;padding:8px 16px;font-size:16px">Toggle Depth</button>
<img src='/feed' style='max-width:100%'>
</body></html>"""

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
