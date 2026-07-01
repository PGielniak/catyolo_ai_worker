import time
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from detector.scene_registry import ScenePipelineRegistry


def run_stream(registry: ScenePipelineRegistry, port: int = 5001):
    """Serve a single FastAPI app exposing every scene's MJPEG + health on
    one port. Routes are per-scene: /feed/{scene_id}, /healthz/{scene_id},
    /toggle_depth/{scene_id}, /depth_tuning/{scene_id}. /healthz reports
    aggregate status."""

    app = FastAPI()
    # The React frontend (served by the backend, different port) needs to call
    # the live depth-tuning endpoints here directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _runner(scene_id: str):
        """Return the scene's inference backend (Hailo10Backend) or None."""
        entry = registry.get(scene_id)
        if entry is None:
            return None
        pipeline, _capture = entry
        with pipeline._config_lock:
            return pipeline._hailo_runner

    def _frames(scene_id: str):
        entry = registry.get(scene_id)
        if entry is None:
            return
        pipeline, _capture = entry
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

    @app.get("/healthz")
    def healthz():
        entries = registry.items()
        if not entries:
            return JSONResponse({"status": "no_scenes", "scenes": []}, status_code=503)
        scene_states = []
        all_ok = True
        for scene_id, pipeline, capture in entries:
            capture_alive = capture.is_alive()
            frame_age = capture.last_frame_age()
            with pipeline._config_lock:
                hailo = pipeline._hailo_runner
            hailo_ready = hailo.wait_until_ready(timeout=0.0) if hailo is not None else False
            ok = capture_alive and hailo_ready
            all_ok = all_ok and ok
            scene_states.append({
                "scene_id": scene_id,
                "status": "ok" if ok else "degraded",
                "capture": capture_alive,
                "hailo": hailo_ready,
                "last_frame_age_s": round(frame_age, 2) if frame_age is not None else None,
            })
        return JSONResponse(
            {"status": "ok" if all_ok else "degraded", "scenes": scene_states},
            status_code=200 if all_ok else 503,
        )

    @app.get("/healthz/{scene_id}")
    def healthz_scene(scene_id: str):
        entry = registry.get(scene_id)
        if entry is None:
            return JSONResponse({"detail": f"unknown scene {scene_id}"}, status_code=404)
        pipeline, capture = entry
        capture_alive = capture.is_alive()
        frame_age = capture.last_frame_age()
        with pipeline._config_lock:
            hailo = pipeline._hailo_runner
        hailo_ready = hailo.wait_until_ready(timeout=0.0) if hailo is not None else False
        status = "ok" if (capture_alive and hailo_ready) else "degraded"
        code = 200 if status == "ok" else 503
        return JSONResponse({
            "scene_id": scene_id,
            "status": status,
            "capture": capture_alive,
            "hailo": hailo_ready,
            "last_frame_age_s": round(frame_age, 2) if frame_age is not None else None,
        }, status_code=code)

    @app.get("/toggle_depth/{scene_id}")
    def toggle_depth(scene_id: str):
        entry = registry.get(scene_id)
        if entry is None:
            return JSONResponse({"detail": f"unknown scene {scene_id}"}, status_code=404)
        pipeline, _capture = entry
        current = pipeline.get_depth_show()
        pipeline.set_depth_show(not current)
        return {"scene_id": scene_id, "depth_show": not current}

    class DepthTuningRequest(BaseModel):
        depth_diff_threshold: float | None = None
        depth_diff_downsample: int | None = None
        depth_smooth_window: int | None = None
        depth_guided_radius: int | None = None
        depth_guided_eps: float | None = None

    @app.get("/depth_tuning/{scene_id}")
    def get_depth_tuning(scene_id: str):
        runner = _runner(scene_id)
        if runner is None:
            return JSONResponse({"detail": f"unknown scene {scene_id}"}, status_code=404)
        if not runner.capabilities.supports_depth:
            return JSONResponse({"detail": "depth not supported on this backend"}, status_code=400)
        return {"scene_id": scene_id, "tuning": runner.get_depth_tuning()}

    @app.post("/depth_tuning/{scene_id}")
    def set_depth_tuning(scene_id: str, body: DepthTuningRequest):
        runner = _runner(scene_id)
        if runner is None:
            return JSONResponse({"detail": f"unknown scene {scene_id}"}, status_code=404)
        if not runner.capabilities.supports_depth:
            return JSONResponse({"detail": "depth not supported on this backend"}, status_code=400)
        params = {k: v for k, v in body.model_dump().items() if v is not None}
        applied = runner.set_depth_tuning(params)
        return {"scene_id": scene_id, "applied": applied, "tuning": runner.get_depth_tuning()}

    @app.get("/feed/{scene_id}")
    def feed(scene_id: str):
        entry = registry.get(scene_id)
        if entry is None:
            return JSONResponse({"detail": f"unknown scene {scene_id}"}, status_code=404)
        return StreamingResponse(_frames(scene_id), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/", response_class=HTMLResponse)
    def index():
        scene_ids = registry.scene_ids()
        if not scene_ids:
            return "<html><body style='margin:0;background:#111;color:#fff;font-family:monospace;padding:20px'>No scenes running.</body></html>"
        links = "".join(
            f"<div style='margin:8px'><a href='/feed/{sid}' style='color:#6cf'>{sid}</a> "
            f"<a href='/healthz/{sid}' style='color:#999;font-size:12px'>health</a></div>"
            for sid in scene_ids
        )
        return f"""<html><body style="margin:0;background:#111;color:#fff;font-family:monospace;padding:20px">
<h2>CatYolo scenes ({len(scene_ids)})</h2>
{links}
</body></html>"""

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
