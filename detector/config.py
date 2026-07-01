import base64
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_rtsp_url(scene: dict) -> str:
    """Build the RTSP URL from the scene's camera fields.

    Mirrors the backend convention in routes/frame.py:
      rtsp://<user>:<pass>@<ip>:<port>/stream1
    Credentials are percent-encoded with safe='' so a ':' or '@' in the
    password can't corrupt the URL. This is the single source of truth for
    the camera URL now that the worker derives it from the scene (not from
    an RTSP_URL env var).
    """
    ip = str(scene.get("camera_ip_address") or "").strip()
    port = scene.get("camera_port") or ""
    user = quote(str(scene.get("camera_username") or ""), safe="")
    pw = quote(str(scene.get("camera_password") or ""), safe="")
    if not ip:
        return ""
    return f"rtsp://{user}:{pw}@{ip}:{port}/stream1"


def _decode_reference_image(b64: Optional[str]) -> Optional[np.ndarray]:
    """Decode a base64-encoded JPEG/PNG from the backend into a BGR numpy array."""
    if not b64:
        return None
    try:
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cv2.imdecode returned None for reference image")
        return img
    except Exception:
        logger.exception("Failed to decode reference image")
        return None


def _union_classes(red_zones: list[dict]) -> list[str]:
    classes: set[str] = set()
    for rz in red_zones or []:
        for c in rz.get("forbidden_classes") or []:
            classes.add(c)
    return list(classes) if classes else ["cat"]


def _normalize_zone(zone: dict) -> dict:
    """Ensure every red zone has both a `points` polygon and x/y/width/height.

    New zones ship `points`; legacy zones only have x/y/width/height.
    Back-fill whichever is missing so polygon-aware code and rectangle-only
    code (e.g. the occlusion detector's bounding-box approximation) both work.
    """
    zone = dict(zone)
    raw_points = zone.get("points")
    has_points = raw_points and len(raw_points) >= 3
    x = zone.get("x")
    y = zone.get("y")
    w = zone.get("width")
    h = zone.get("height")
    has_rect = x is not None and y is not None and w is not None and h is not None

    if has_points and has_rect:
        return zone

    if has_points:
        poly = [(float(p[0]), float(p[1])) for p in raw_points]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))
        zone["x"] = x1
        zone["y"] = y1
        zone["width"] = x2 - x1
        zone["height"] = y2 - y1
        return zone

    # Legacy rectangle -> points
    x = int(zone.get("x") or 0)
    y = int(zone.get("y") or 0)
    w = int(zone.get("width") or 0)
    h = int(zone.get("height") or 0)
    zone["points"] = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return zone


@dataclass(frozen=True)
class SceneConfig:
    """Immutable snapshot of the scene config the pipeline is currently running
    against. Replacement, not mutation: when the backend scene changes, the
    pipeline builds a brand new SceneConfig and atomically swaps it in.

    The `red_zones` list contains *fresh* dict copies (not references to the
    raw backend payload), so the occlusion detector can safely mutate per-zone
    `x`/`y` for template tracking without leaking state across reloads.
    """

    scene: dict
    reference_image: Optional[np.ndarray]
    red_zones: list[dict] = field(default_factory=list)
    scene_prompt: Optional[str] = None
    scene_prompt_interval: Optional[int] = None
    scene_prompt_action_ids: Optional[list[str]] = None
    global_detection_enabled: bool = False
    global_detection_classes: list[str] = field(default_factory=list)
    global_detection_action_ids: Optional[list[str]] = None
    global_detection_cooldown_seconds: int = 60
    forbidden_classes: list[str] = field(default_factory=lambda: ["cat"])
    version: int = 0
    # RTSP URL built from the scene's camera_* fields (single source of
    # truth; supersedes the old RTSP_URL env var). Empty when the scene has
    # no camera_ip_address.
    rtsp_url: str = ""

    @classmethod
    def from_scene_dict(cls, scene: dict) -> "SceneConfig":
        """Build a SceneConfig from a raw scene dict (as returned by the backend
        on GET /scene/internal/). Deep-copies the red zone dicts so downstream
        mutators (occlusion tracking) can't trample later payloads."""
        raw_zones = scene.get("red_zones") or []
        red_zones = [_normalize_zone(dict(rz)) for rz in raw_zones]
        ref_img = _decode_reference_image((scene.get("image") or {}).get("image"))
        return cls(
            scene=dict(scene),
            reference_image=ref_img,
            red_zones=red_zones,
            scene_prompt=scene.get("scene_prompt"),
            scene_prompt_interval=scene.get("scene_prompt_interval"),
            scene_prompt_action_ids=copy.deepcopy(scene.get("scene_prompt_action_ids")),
            global_detection_enabled=bool(scene.get("global_detection_enabled")),
            global_detection_classes=list(scene.get("global_detection_classes") or []),
            global_detection_action_ids=copy.deepcopy(scene.get("global_detection_action_ids")),
            global_detection_cooldown_seconds=int(scene.get("global_detection_cooldown_seconds") or 60),
            forbidden_classes=_union_classes(raw_zones),
            version=int(scene.get("version") or 0),
            rtsp_url=build_rtsp_url(scene),
        )
