# catyolo-worker

Detection worker for the CatYolo system. Reads RTSP frames from a fixed IP camera,
runs the detection pipeline, and posts alerts back to the CRUD API.

## Projects in this system

| Project | Port | Purpose |
|---|---|---|
| `catyolo_ai_backend` | 8100 | CRUD API — scenes, zones, actions, logs |
| `catyolo_ai_worker` | 5001 (optional) | Detection pipeline + debug video stream |

Both share the same SQLite database file (path configured via `.env`).

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- OpenCV-compatible RTSP camera (tested with TP-Link Tapo)
- Raspberry Pi 5 + Hailo-10H AI HAT+ 2 (for YOLO/VLM stages — not required for raw stream)

## Setup

```bash
cp .env.example .env
# edit .env with your camera URL and API base
```

## Running

```bash
~/.local/bin/uv run worker
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | — | Full RTSP URL including credentials |
| `API_BASE` | `http://localhost:8100` | Base URL of `catyolo_ai_backend` |
| `ENABLE_STREAM` | `false` | Set to `true` to enable debug MJPEG stream on port 5001 |
| `STREAM_PORT` | `5001` | Port for the debug stream |

## Debug video stream

Set `ENABLE_STREAM=true` then open `http://raspberry01.local:5001` in a browser.
The stream serves the raw latest frame at ~15 fps as MJPEG.

Disable when not debugging — it encodes every frame and wastes CPU.

## Project structure

```
detector/
├── capture.py    # FrameCapture — background thread, single-slot RTSP buffer
├── pipeline.py   # DetectionPipeline — main loop, fetches config, runs stages
├── stream.py     # Optional MJPEG debug feed (FastAPI on STREAM_PORT)
└── main.py       # Entrypoint — wires everything together, reads .env
```

## Detection pipeline stages (planned)

1. **Occlusion check** (CPU) — SSIM/IoU vs reference frame per red zone
2. **YOLO cat detection** (Hailo NPU) — runs in parallel with stage 1
3. **Zone overlap** — does detected cat bbox overlap a clear red zone?
4. **Depth check** — is the cat at the same depth as the plant?
5. **VLM confirmation** (Hailo NPU) — "is the cat attacking the plant?"

Stages 1–2 run on every frame. Stages 3–5 run only when earlier stages pass.
YOLO and VLM share one Hailo inference context — access is serialized via a lock.
