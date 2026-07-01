import cv2
import numpy as np

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]


def letterbox(
    image: np.ndarray,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """Resize image to fit inside target_h × target_w with black padding.

    Returns (padded_image, top, bottom, left, right) where top/bottom are vertical
    pad amounts and left/right are horizontal pad amounts.
    """
    orig_h, orig_w = image.shape[:2]
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    v_pad = target_h - new_h  # total vertical (height) padding
    h_pad = target_w - new_w  # total horizontal (width) padding

    top = v_pad // 2
    bottom = v_pad - top
    left = h_pad // 2
    right = h_pad - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )
    return padded, top, bottom, left, right


def bbox_unmap(
    x1_n: float,
    y1_n: float,
    x2_n: float,
    y2_n: float,
    desired_h: int,
    desired_w: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    orig_h: int,
    orig_w: int,
) -> tuple[int, int, int, int]:
    """Map normalised bbox coords from a letterboxed canvas back to original pixel space.

    Coords are normalised in [0, 1] relative to the desired_h × desired_w canvas.
    Returns (x1, y1, x2, y2) in the orig_h × orig_w image.
    """
    x1 = int(x1_n * desired_w) - left
    y1 = int(y1_n * desired_h) - top
    x2 = int(x2_n * desired_w) - left
    y2 = int(y2_n * desired_h) - top

    eff_w = desired_w - left - right
    eff_h = desired_h - top - bottom

    x1 = int(x1 * orig_w / eff_w)
    y1 = int(y1 * orig_h / eff_h)
    x2 = int(x2 * orig_w / eff_w)
    y2 = int(y2 * orig_h / eff_h)
    return x1, y1, x2, y2
