"""Unit tests for bbox coordinate transform — no NPU needed.

Covers preprocessing.letterbox() + preprocessing.bbox_unmap() directly.
"""
from detector.inference.preprocessing import bbox_unmap, letterbox
import numpy as np


def test_square_desired_no_padding():
    """640x640 frame into 640x640 desired: no padding, coords are identity."""
    x1, y1, x2, y2 = bbox_unmap(
        x1_n=0.25, y1_n=0.25, x2_n=0.75, y2_n=0.75,
        desired_h=640, desired_w=640,
        top=0, bottom=0, left=0, right=0,
        orig_h=640, orig_w=640,
    )
    assert x1 == 160
    assert y1 == 160
    assert x2 == 480
    assert y2 == 480


def test_asymmetric_desired_exposes_axis_swap():
    """Non-square HEF (480h x 640w) with no padding: x must use desired_w, y must use desired_h.

    The OLD (buggy) code used desired_h for x and desired_w for y, which gave:
      x1 = int(0.5 * 480) = 240  (wrong)
      y1 = int(0.5 * 640) = 320  (wrong)
    The FIXED code gives:
      x1 = int(0.5 * 640) = 320  (correct centre-x)
      y1 = int(0.5 * 480) = 240  (correct centre-y)
    """
    x1, y1, x2, y2 = bbox_unmap(
        x1_n=0.5, y1_n=0.5, x2_n=0.5, y2_n=0.5,
        desired_h=480, desired_w=640,
        top=0, bottom=0, left=0, right=0,
        orig_h=480, orig_w=640,
    )
    assert x1 == 320, f"expected x1=320, got {x1} (axis-swap bug?)"
    assert y1 == 240, f"expected y1=240, got {y1} (axis-swap bug?)"


def test_letterbox_vertical_padding():
    """480x640 frame letterboxed into 640x640: top/bottom pads of 80px each."""
    x1, y1, x2, y2 = bbox_unmap(
        x1_n=0.5, y1_n=0.5, x2_n=0.5, y2_n=0.5,
        desired_h=640, desired_w=640,
        top=80, bottom=80, left=0, right=0,
        orig_h=480, orig_w=640,
    )
    assert x1 == 320, f"expected x1=320 (centre x), got {x1}"
    assert y1 == 240, f"expected y1=240 (centre y), got {y1}"


def test_detection_at_top_left_corner():
    """Detection at top-left of original image (after padding removal)."""
    x1, y1, x2, y2 = bbox_unmap(
        x1_n=0.0, y1_n=0.125, x2_n=0.0, y2_n=0.125,
        desired_h=640, desired_w=640,
        top=80, bottom=80, left=0, right=0,
        orig_h=480, orig_w=640,
    )
    assert x1 == 0
    assert y1 == 0


def test_full_image_detection_maps_to_original_bounds():
    """A detection spanning the full padded image maps back to full original."""
    x1, y1, x2, y2 = bbox_unmap(
        x1_n=0.0, y1_n=0.125,
        x2_n=1.0, y2_n=0.875,
        desired_h=640, desired_w=640,
        top=80, bottom=80, left=0, right=0,
        orig_h=480, orig_w=640,
    )
    assert x1 == 0
    assert y1 == 0
    assert x2 == 640
    assert y2 == 480


def test_letterbox_output_shape():
    """letterbox() returns an image of exactly the requested dimensions."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, top, bottom, left, right = letterbox(img, 640, 640)
    assert padded.shape == (640, 640, 3)
    assert top + bottom == 160  # 640 - 480
    assert left == 0
    assert right == 0


def test_letterbox_wide_image():
    """Wider-than-tall image (e.g. 360x640) letterboxed into 640x640."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    padded, top, bottom, left, right = letterbox(img, 640, 640)
    assert padded.shape == (640, 640, 3)
    assert left == 0 and right == 0  # no horizontal pad needed
    assert top + bottom == 280       # 640 - 360
