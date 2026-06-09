import base64
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
    forbidden_classes: list[str] = field(default_factory=lambda: ["cat"])
    version: int = 0

    @classmethod
    def from_scene_dict(cls, scene: dict) -> "SceneConfig":
        """Build a SceneConfig from a raw scene dict (as returned by the backend
        on GET /scene/). Deep-copies the red zone dicts so downstream mutators
        (occlusion tracking) can't trample later payloads."""
        raw_zones = scene.get("red_zones") or []
        red_zones = [dict(rz) for rz in raw_zones]
        ref_img = _decode_reference_image((scene.get("image") or {}).get("image"))
        return cls(
            scene=dict(scene),
            reference_image=ref_img,
            red_zones=red_zones,
            scene_prompt=scene.get("scene_prompt"),
            scene_prompt_interval=scene.get("scene_prompt_interval"),
            scene_prompt_action_ids=copy.deepcopy(scene.get("scene_prompt_action_ids")),
            forbidden_classes=_union_classes(raw_zones),
            version=int(scene.get("version") or 0),
        )
