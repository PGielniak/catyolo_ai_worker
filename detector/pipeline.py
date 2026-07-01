import logging
import os
import time
import threading
import cv2
logger = logging.getLogger(__name__)
import numpy as np
from datetime import datetime
from typing import Any
from detector.detectors.occlusion_detectionV2 import OcclusionDetector
from detector.inference.factory import create_backend
from detector.inference.protocols import InferenceBackend
from detector.events import DetectionEventEmitter, DetectionEvent, dispatch_event
from detector.config import SceneConfig
from detector.geometry import (
    box_polygon_intersection_area,
    point_in_polygon,
    polygon_bounding_box,
    zone_to_polygon,
)

import ctypes

def set_thread_name(name: str):
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass
class DetectionPipeline:
    VLM_COOLDOWN_SECONDS = 3
    VLM_MIN_OVERLAP_SECONDS = 1.0
    DEPTH_MARGIN_DEFAULT = 0.20
    # How long to wait for occlusion / hailo threads to stop on reload.
    RELOAD_STOP_TIMEOUT = 5.0
    # Settle and setup timeouts are now backend-specific; read from the backend
    # instance via RELOAD_SETTLE_SECONDS / SETUP_TIMEOUT class attributes.

    def __init__(self, capture, api_base: str, initial_config: SceneConfig, shared_device: Any = None):
        self._capture = capture
        self._api_base = api_base
        # Shared Hailo VDevice (multi-camera). When None, the backend owns its
        # own device (legacy/test path). Injected so all per-scene pipelines
        # multiplex over one physical device via the HailoRT ROUND_ROBIN scheduler.
        self._shared_device = shared_device
        self._annotated = None
        self._annotated_lock = threading.Lock()
        self._depth_viz = None
        self._depth_viz_lock = threading.Lock()
        self._depth_show = True
        self._depth_show_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # --- single source of truth for live config ---------------------------
        # _current_config is a frozen SceneConfig snapshot. The hot loop reads
        # it once per frame under _config_lock and uses the snapshot for the
        # rest of the iteration. reload_config() swaps it atomically.
        self._config_lock = threading.Lock()
        self._current_config: SceneConfig = initial_config
        # Pre-computed OpenCV polygon arrays so _draw_zones doesn't rebuild them
        # every frame. Recomputed whenever the config reloads.
        self._red_zone_polygons = self._compute_zone_polygons(initial_config.red_zones)
        # Serializes reload_config() calls so two reloads can't race.
        self._reload_lock = threading.Lock()

        # Per-zone throttling state — keyed by zone index. Lives on the
        # pipeline (not the config) because it's runtime state, not config.
        # Cleared on every reload so old indices don't leak into new zones.
        self._last_event_by_zone = {}
        self._overlap_since = {}
        self._vlm_fired_for_zone = set()

        # VLM result bookkeeping
        self._last_vlm_answer = None
        self._last_vlm_answer_zone = None
        self._last_vlm_prompt = None
        self._last_vlm_frame: np.ndarray | None = None
        self._last_vlm_answer_lock = threading.Lock()
        self._last_processed_vlm_result = None

        # Global-scene-prompt bookkeeping
        self._last_global_vlm_time = 0.0
        self._pending_vlm_is_global = False

        # Global object-detection trigger bookkeeping
        self._last_global_detection_time = 0.0

        # Reference depths — read from the hailo runner as it finishes
        # computing them.
        self._latest_occlusion_result = None
        self._reference_depths_ok = False
        self._reference_depths: dict[int, float] = {}

        # Build the first set of detectors from the initial config
        self._occlusion_detector = self._build_occlusion(initial_config)
        self._hailo_runner = self._build_hailo(initial_config)

        self._detection_events = DetectionEventEmitter()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_occlusion(self, cfg: SceneConfig) -> OcclusionDetector:
        if cfg.reference_image is None:
            raise RuntimeError("Initial scene config has no reference image; cannot build OcclusionDetector")
        return OcclusionDetector(
            capture=self._capture,
            reference_image=cfg.reference_image,
            red_zones=cfg.red_zones,
        )

    def _build_hailo(self, cfg: SceneConfig) -> InferenceBackend:
        return create_backend(
            capture=self._capture,
            yolo_classes=cfg.forbidden_classes,
            reference_image=cfg.reference_image,
            red_zones=cfg.red_zones,
            shared_device=self._shared_device,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def subscribe(self, handler):
        self._detection_events.subscribe(handler)

    def start(self):
        self._thread.start()
        self._hailo_runner.start()
        # The occlusion detector is driven inline from the pipeline thread
        # (see _run -> occlusion.process(frame)). Its own background thread
        # is not used in the current architecture, so we don't call .start()
        # on it.

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        # Stop the Hailo backend too so a per-scene shutdown (multi-camera)
        # doesn't leave its hailo thread running. With a shared device the
        # backend's _teardown releases only its model handles, NOT the shared
        # VDevice — so stopping one scene never pulls the device out from under
        # the other scenes.
        try:
            if self._hailo_runner is not None:
                self._hailo_runner.stop(timeout=self.RELOAD_STOP_TIMEOUT)
        except Exception:
            logger.exception("Error stopping HailoRunner during pipeline shutdown")

    def reload_config(self, new_config: SceneConfig):
        """Atomically replace the running config, rebuilding the occlusion
        detector and the Hailo runner so the new reference image, red zone
        geometry, and forbidden-class set all take effect.

        Safe to call from any thread. Concurrent reloads are serialized; the
        in-flight reload blocks until this one completes. Reloads are also
        safe to call before start() — in that case we just swap the
        references and the next start() will use the new instances.
        """
        # Serialize concurrent reload requests. If another reload is in flight,
        # just drop this one — the in-flight one will pick up the same new
        # config (or a newer one on the next tick).
        if not self._reload_lock.acquire(blocking=False):
            logger.debug("Reload already in progress; skipping this tick")
            return
        try:
            old_version = self._current_config.version
            logger.info(
                "Reloading scene config — version %s -> %s, scene_id=%s, zones=%d, classes=%s",
                old_version,
                new_config.version,
                new_config.scene.get("scene_id"),
                len(new_config.red_zones),
                new_config.forbidden_classes,
            )

            if new_config.reference_image is None:
                logger.warning(
                    "Refusing to reload: new config has no reference image (scene_id=%s)",
                    new_config.scene.get("scene_id"),
                )
                return

            # ---- Tear down old detectors ----
            try:
                if self._hailo_runner is not None:
                    self._hailo_runner.stop(timeout=self.RELOAD_STOP_TIMEOUT)
            except Exception:
                logger.exception("Error stopping HailoRunner during reload")
            try:
                if self._occlusion_detector is not None:
                    self._occlusion_detector.stop(timeout=self.RELOAD_STOP_TIMEOUT)
            except Exception:
                logger.exception("Error stopping OcclusionDetector during reload")

            # HailoRT needs a brief settle window after VDevice.release() before
            # a new VDevice can be opened on the same physical device. Without
            # this the new runner's _setup() fails with
            # HAILO_DEVICE_TEMPERARILY_UNAVAILABLE(97).
            _settle = getattr(self._hailo_runner, "RELOAD_SETTLE_SECONDS", 2.0)
            if _settle > 0:
                logger.info("Settling for %.1fs before re-opening Hailo device", _settle)
                time.sleep(_settle)

            # ---- Build new detectors ----
            try:
                new_occlusion = self._build_occlusion(new_config)
                new_hailo = self._build_hailo(new_config)
            except Exception:
                logger.exception("Failed to build new detectors; aborting reload")
                return

            # ---- Reset per-zone throttling so old indices don't bleed in ----
            self._last_event_by_zone = {}
            self._overlap_since = {}
            self._vlm_fired_for_zone = set()
            self._reference_depths = {}
            self._reference_depths_ok = False
            self._last_processed_vlm_result = None
            with self._last_vlm_answer_lock:
                self._last_vlm_answer = None
            self._last_vlm_answer_zone = None
            self._last_vlm_prompt = None
            self._last_vlm_frame = None
            self._pending_vlm_is_global = False
            self._last_global_detection_time = 0.0

            # ---- Atomic swap ----
            with self._config_lock:
                self._current_config = new_config
                self._occlusion_detector = new_occlusion
                self._hailo_runner = new_hailo
                self._red_zone_polygons = self._compute_zone_polygons(new_config.red_zones)

            # ---- Start new threads (no-op if start() was never called) ----
            try:
                new_hailo.start()
            except Exception:
                logger.exception("Error starting new HailoRunner")
            # Occlusion detector is driven inline; no background thread to start.

            # Wait for the new runner to actually finish binding to the NPU
            # before declaring the reload complete. If setup fails, log it and
            # back out so the user sees a real error instead of silent no-detection.
            _setup_timeout = getattr(new_hailo, "SETUP_TIMEOUT", 60.0)
            ready = new_hailo.wait_until_ready(timeout=_setup_timeout)
            if not ready:
                logger.error(
                    "Hailo backend did not finish setup within %.0fs after reload; "
                    "detection will not run until the worker is restarted",
                    _setup_timeout,
                )

            logger.info(
                "Scene config reload complete — version=%s zones=%d classes=%s",
                new_config.version,
                len(new_config.red_zones),
                new_config.forbidden_classes,
            )
        finally:
            self._reload_lock.release()

    def set_depth_show(self, enabled: bool):
        with self._depth_show_lock:
            self._depth_show = enabled

    def get_depth_show(self) -> bool:
        with self._depth_show_lock:
            return self._depth_show

    def get_annotated(self):
        with self._annotated_lock:
            return self._annotated.copy() if self._annotated is not None else None

    def get_depth_viz(self):
        with self._depth_viz_lock:
            return self._depth_viz.copy() if self._depth_viz is not None else None

    def get_last_vlm_answer(self):
        with self._last_vlm_answer_lock:
            return self._last_vlm_answer

    @staticmethod
    def _box_polygon_overlap(box, polygon):
        return box_polygon_intersection_area(box, polygon) > 0

    def _overlapping_zone_indices(self, yolo_result, red_zones):
        """Returns {zone_index: first_matching_class} for zones with a forbidden-class bbox overlap."""
        if yolo_result is None or not yolo_result.detections:
            return {}
        hits = {}
        for det in yolo_result.detections:
            det_box = (det.x1, det.y1, det.x2, det.y2)
            for idx, rz in enumerate(red_zones):
                if idx in hits:
                    continue
                poly = zone_to_polygon(rz)
                if self._box_polygon_overlap(det_box, poly):
                    forbidden = rz.get("forbidden_classes", [])
                    if det.label in forbidden:
                        hits[idx] = det.label
        return hits

    def _check_global_detection(self, yolo_result, cfg: SceneConfig, now_mono: float):
        """Emit a global detection event if a configured class is seen anywhere in the frame
        and the cooldown has elapsed. Returns the matching class or None."""
        if not cfg.global_detection_enabled:
            return None
        trigger_classes = cfg.global_detection_classes or []
        if not trigger_classes:
            return None
        if yolo_result is None or not yolo_result.detections:
            return None
        cooldown = cfg.global_detection_cooldown_seconds or 60
        if now_mono - self._last_global_detection_time < cooldown:
            return None
        for det in yolo_result.detections:
            if det.label in trigger_classes:
                return det.label
        return None

    def _any_zone_wants_depth(self, red_zones) -> bool:
        for rz in red_zones:
            if rz.get("depth_enabled"):
                return True
        return False

    def _check_depth_match(self, zone_idx: int, red_zones, depth_map, detection_bbox) -> bool:
        if not self._reference_depths_ok or zone_idx not in self._reference_depths:
            return True
        ref_depth = self._reference_depths[zone_idx]
        if zone_idx >= len(red_zones):
            return True
        zone = red_zones[zone_idx]
        margin = zone.get("depth_margin") or self.DEPTH_MARGIN_DEFAULT

        x1 = max(0, int(detection_bbox[0]))
        y1 = max(0, int(detection_bbox[1]))
        x2 = min(depth_map.shape[1], int(detection_bbox[2]))
        y2 = min(depth_map.shape[0], int(detection_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return True
        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            return True
        bbox_median = float(np.median(crop))
        diff = abs(bbox_median - ref_depth) / max(abs(ref_depth), 1e-8)
        if diff > margin:
            logger.info(f"Depth gate blocked — zone {zone_idx}: "
                        f"ref={ref_depth:.3f} bbox={bbox_median:.3f} diff={diff:.3f} > margin={margin:.3f}")
            return False
        return True

    def _run(self):
        set_thread_name("pipeline")
        logger.debug(f"Entered _run method in DetectionPipeline")
        with self._config_lock:
            initial = self._current_config
        logger.debug(f"{initial.scene.get('scene_id')}")
        logger.debug(f"{initial.scene.get('scene_name')}")
        logger.debug(f"{initial.scene.get('camera_ip_address')}")
        logger.debug(f"{initial.scene.get('camera_port')}")
        logger.debug(f"{initial.red_zones}")
        # Per-scene FPS cap. With multi-camera (WS2) the HailoRT ROUND_ROBIN
        # scheduler time-slices the NPU across scenes; the single-slot capture
        # buffer drops to newest for free when the NPU can't keep up. Lowering
        # TARGET_FPS reduces wasted CPU occlusion work when the NPU is the
        # bottleneck. Tunable via env so multi-camera deployments can dial it
        # down (e.g. 3 feeds at ~10fps each) without a code change.
        TARGET_FPS = int(os.getenv("TARGET_FPS", "30"))
        target_dt = 1.0 / TARGET_FPS
        next_tick = time.monotonic()

        fps_times = []

        while not self._stop_event.is_set():
            frame = self._capture.get()
            if frame is None:
                time.sleep(0.05)
                continue

            annotated = frame.copy()

            # Snapshot the live config for this frame. The lock acquire is
            # very short (one attribute read), so contention is negligible.
            with self._config_lock:
                cfg = self._current_config
                occlusion = self._occlusion_detector
                hailo = self._hailo_runner
                zone_polys = self._red_zone_polygons

            red_zones = cfg.red_zones
            scene_prompt = cfg.scene_prompt
            scene_prompt_interval = cfg.scene_prompt_interval

            self._draw_zones(annotated, zone_polys)

            result = occlusion.process(frame)

            if result is not None:
                self._latest_occlusion_result = result
                self._draw_occlusion(annotated, result)

            yolo_detection = hailo.get_latest()

            if yolo_detection is not None:
                self._draw_yolo_detection(annotated, yolo_detection)

                if yolo_detection.vlm_answer is not None and yolo_detection is not self._last_processed_vlm_result:
                    self._last_processed_vlm_result = yolo_detection
                    now_vlm = datetime.now()
                    answer = yolo_detection.vlm_answer
                    with self._last_vlm_answer_lock:
                        self._last_vlm_answer = (answer, now_vlm)
                    logger.info(f"VLM result received: '{answer}'")
                    self._draw_vlm_answer(annotated, answer, self._last_vlm_answer_zone, question=self._last_vlm_prompt)

                    if self._pending_vlm_is_global:
                        self._pending_vlm_is_global = False
                        global_event = DetectionEvent(
                            annotated_image=annotated.copy(),
                            trigger="global_description",
                            raw_frame=self._last_vlm_frame.copy() if self._last_vlm_frame is not None else None,
                            vlm_prompt=self._last_vlm_prompt or "",
                            vlm_answer=answer,
                            timestamp=now_vlm,
                            zone=None,
                            is_global_prompt=True,
                            scene_id=cfg.scene.get("scene_id"),
                        )
                        self._detection_events.emit(global_event)
                        with self._config_lock:
                            global_action_ids = self._current_config.scene_prompt_action_ids or []
                        dispatch_event(global_event, global_action_ids)
                    elif self._last_vlm_answer_zone is not None:
                        zone = self._last_vlm_answer_zone
                        vlm_decides_trigger = zone.get("vlm_decides_trigger")
                        # Default (None/False): VLM is informational only — always fire.
                        # True: legacy behaviour — fire only on a Yes answer.
                        if vlm_decides_trigger is True and "yes" not in answer.lower():
                            pass
                        else:
                            det_class = zone.get("forbidden_classes", [])[0] if zone.get("forbidden_classes") else None
                            trigger_name = "vlm_yes" if vlm_decides_trigger is True else "vlm_processed"
                            vlm_event = DetectionEvent(
                                annotated_image=annotated.copy(),
                                trigger=trigger_name,
                                raw_frame=self._last_vlm_frame.copy() if self._last_vlm_frame is not None else None,
                                detected_class=det_class,
                                vlm_prompt=self._last_vlm_prompt or "",
                                vlm_answer=answer,
                                timestamp=now_vlm,
                                zone=zone,
                                scene_id=cfg.scene.get("scene_id"),
                            )
                            self._detection_events.emit(vlm_event)
                            zone_action_ids = zone.get("action_ids") or []
                            dispatch_event(vlm_event, zone_action_ids)

                if yolo_detection.yolo_result is not None:
                    depth_on = self.get_depth_show() or self._any_zone_wants_depth(red_zones)
                    hailo.set_depth_enabled(depth_on)

                    if not self._reference_depths_ok:
                        ready, ref_depths = hailo.get_reference_depths(timeout=0.0)
                        if ready:
                            self._reference_depths = ref_depths
                            self._reference_depths_ok = True

                    now_vlm = time.monotonic()
                    overlapping_now = self._overlapping_zone_indices(yolo_detection.yolo_result, red_zones)

                    for zi in list(self._overlap_since):
                        if zi not in overlapping_now:
                            self._overlap_since.pop(zi, None)
                            self._vlm_fired_for_zone.discard(zi)

                    for zi, detected_class in overlapping_now.items():
                        if zi not in self._overlap_since:
                            self._overlap_since[zi] = now_vlm
                        elif zi not in self._vlm_fired_for_zone:
                            elapsed = now_vlm - self._overlap_since[zi]
                            if elapsed >= self.VLM_MIN_OVERLAP_SECONDS:
                                last_request = self._last_event_by_zone.get(zi)
                                if last_request is None or (now_vlm - last_request) >= self.VLM_COOLDOWN_SECONDS:
                                    zone = red_zones[zi]

                                    zone_occluded = False
                                    if (self._latest_occlusion_result is not None and
                                        zi < len(self._latest_occlusion_result.zones)):
                                        rz = self._latest_occlusion_result.zones[zi]
                                        if rz.get("occluded"):
                                            zone_occluded = True
                                            logger.info(f"Occlusion gate blocked — zone {zi} is occluded "
                                                        f"(score={rz.get('occlusion_score', 0):.2f})")
                                    if zone_occluded:
                                        continue

                                    if not zone.get("vlm_prompt") or not hailo.capabilities.supports_vlm:
                                        if zone.get("depth_enabled") and yolo_detection.depth_map is not None:
                                            det_box = None
                                            for det in yolo_detection.yolo_result.detections:
                                                if det.label == detected_class:
                                                    det_box = (det.x1, det.y1, det.x2, det.y2)
                                                    break
                                            if det_box and not self._check_depth_match(zi, red_zones, yolo_detection.depth_map, det_box):
                                                continue
                                            trigger = "depth_match"
                                        else:
                                            trigger = "overlap"
                                        logger.info(f"DetectionEvent trigger — zone {zi} overlapped for {elapsed:.1f}s, class={detected_class}, trigger={trigger}")
                                        zone_event = DetectionEvent(
                                            annotated_image=annotated.copy(),
                                            trigger=trigger,
                                            raw_frame=frame.copy(),
                                            detected_class=detected_class,
                                            timestamp=datetime.now(),
                                            zone=zone,
                                            scene_id=cfg.scene.get("scene_id"),
                                        )
                                        self._detection_events.emit(zone_event)
                                        zone_action_ids = zone.get("action_ids") or []
                                        dispatch_event(zone_event, zone_action_ids)
                                        self._last_event_by_zone[zi] = now_vlm
                                        self._last_vlm_answer_zone = zone
                                        self._last_vlm_prompt = ""
                                        self._vlm_fired_for_zone.add(zi)
                                        self._pending_vlm_is_global = False
                                        continue

                                    if zone.get("depth_enabled") and yolo_detection.depth_map is not None:
                                        det_box = None
                                        for det in yolo_detection.yolo_result.detections:
                                            if det.label == detected_class:
                                                det_box = (det.x1, det.y1, det.x2, det.y2)
                                                break
                                        if det_box and not self._check_depth_match(zi, red_zones, yolo_detection.depth_map, det_box):
                                            continue

                                    prompt_template = zone.get("vlm_prompt") or "Is the {class} attacking a plant?"
                                    prompt = prompt_template.replace("{class}", detected_class)
                                    logger.info(f"VLM trigger — zone {zi} overlapped for {elapsed:.1f}s, class={detected_class}")
                                    hailo.request_vlm(frame, zone=zone, detected_class=detected_class)
                                    self._last_event_by_zone[zi] = now_vlm
                                    self._last_vlm_answer_zone = zone
                                    self._last_vlm_prompt = prompt
                                    self._last_vlm_frame = frame.copy()
                                    self._vlm_fired_for_zone.add(zi)
                                    self._pending_vlm_is_global = False

                    global_class = self._check_global_detection(yolo_detection.yolo_result, cfg, now_vlm)
                    if global_class is not None:
                        logger.info(f"DetectionEvent trigger — global detection, class={global_class}")
                        global_event = DetectionEvent(
                            annotated_image=annotated.copy(),
                            trigger="global_detection",
                            raw_frame=frame.copy(),
                            detected_class=global_class,
                            timestamp=datetime.now(),
                            zone=None,
                            scene_id=cfg.scene.get("scene_id"),
                        )
                        self._detection_events.emit(global_event)
                        global_action_ids = cfg.global_detection_action_ids or []
                        dispatch_event(global_event, global_action_ids)
                        self._last_global_detection_time = now_vlm

                    depth_map = yolo_detection.depth_map
                    if depth_map is not None:
                        min_v, max_v = np.min(depth_map), np.max(depth_map)
                        if max_v > min_v:
                            depth_8u = ((depth_map - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                        else:
                            depth_8u = np.zeros_like(depth_map, dtype=np.uint8)
                        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)
                        with self._depth_viz_lock:
                            self._depth_viz = depth_color

            if (scene_prompt
                    and scene_prompt_interval
                    and scene_prompt_interval > 0
                    and hailo.capabilities.supports_vlm):
                now_mono = time.monotonic()
                if (now_mono - self._last_global_vlm_time) >= scene_prompt_interval:
                    logger.info(f"Global VLM trigger — interval={scene_prompt_interval}s")
                    hailo.request_vlm(
                        frame,
                        zone=None,
                        detected_class=None,
                        is_global=True,
                        global_prompt=scene_prompt,
                    )
                    self._last_global_vlm_time = now_mono
                    self._last_vlm_answer_zone = None
                    self._last_vlm_prompt = scene_prompt
                    self._last_vlm_frame = frame.copy()
                    self._pending_vlm_is_global = True



            now_t = time.monotonic()
            fps_times.append(now_t)
            fps_times[:] = [t for t in fps_times if now_t - t < 1.0]
            fps = len(fps_times)
            cv2.putText(annotated, f"FPS: {fps}", (annotated.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            vlm_display = self.get_last_vlm_answer()
            if vlm_display is not None:
                answer, ts = vlm_display
                self._draw_vlm_answer(annotated, answer, self._last_vlm_answer_zone,
                                      question=self._last_vlm_prompt, color=(255, 255, 255))

            with self._annotated_lock:
                self._annotated = annotated

            next_tick += target_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()


    def _draw_vlm_answer(self, annotated, answer, zone, question=None, color=(255, 255, 255)):
        if zone:
            poly = zone_to_polygon(zone)
            x1, y1, _, _ = polygon_bounding_box(poly)
            x = int(x1)
            y = int(y1)
        else:
            x = 10
            y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        lines = []
        if question:
            lines.append(("Q: " + question, (200, 200, 200)))
        lines.append(("VLM: " + answer, color))

        text_x = x + 4
        text_y = max(y - 22, 14)
        for text, clr in reversed(lines):
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            text_y -= th + 6
            cv2.rectangle(annotated, (text_x - 3, text_y - 2),
                          (text_x + tw + 3, text_y + th + baseline + 2),
                          (0, 0, 0), -1)
            cv2.putText(annotated, text, (text_x, text_y + th),
                        font, scale, clr, thickness)

    @staticmethod
    def _compute_zone_polygons(red_zones):
        polys = []
        for rz in red_zones:
            poly = zone_to_polygon(rz)
            if len(poly) >= 3:
                polys.append(np.array(poly, dtype=np.int32).reshape((-1, 1, 2)))
        return polys

    def _draw_zones(self, annotated, zone_polys):
        if not zone_polys:
            return
        # One overlay, one blend, one outline pass — much cheaper than a
        # per-zone copy/blend when there are several zones.
        overlay = annotated.copy()
        for pts in zone_polys:
            cv2.fillPoly(overlay, [pts], (255, 0, 0))
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        for pts in zone_polys:
            cv2.polylines(annotated, [pts], True, (255, 0, 0), 2)

    def _draw_occlusion(self, annotated, result):
        for rz in result.zones:
            # Draw the configured polygon, not the tracked bounding box.
            # The occlusion math still uses the simple bbox approximation internally.
            poly = zone_to_polygon(rz)
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            x1, y1, _, _ = polygon_bounding_box(poly)

            if rz['occluded']:
                colour = (0, 0, 255)
                status = 'occluded'
            else:
                colour = (0, 165, 255)
                status = 'free'

            cv2.putText(annotated, f"{status} - {rz['occlusion_score']:.2f}",
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            cv2.polylines(annotated, [pts], True, colour, 2, lineType=cv2.LINE_AA)

    def _draw_yolo_detection(self, annotated, detection):
        for det in detection.yolo_result.detections:
            x1 = det.x1
            x2 = det.x2
            y1 = det.y1
            y2 = det.y2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det.label} {det.confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
