import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


CLIP_LIMIT=5
TILE_GRID_SIZE=(6,6)

def _align_to_reference(reference_gray, current_gray):
    shift, response = cv2.phaseCorrelate(
        np.float32(reference_gray), 
        np.float32(current_gray)
    )
    dx, dy = shift
    
    # Sanity check: don't apply huge shifts (likely a bad detection)
    if abs(dx) > 50 or abs(dy) > 50 or response < 0.1:
        return current_gray, (0, 0), response
    
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(
        current_gray, M, 
        (current_gray.shape[1], current_gray.shape[0])
    )
    return aligned, (dx, dy), response


def check_occlusion(frame, reference_image, red_zones):
    ref = _apply_clahe_gray(reference_image)
    compare = _apply_clahe_gray(frame)
    compare_aligned, shift, response = _align_to_reference(ref, compare)
    red_zones_occluded = _compare_clahe_ssim(ref, compare_aligned, red_zones)
    return {
        "zones": red_zones_occluded,
        "shift": shift,
        "alignment_confidence": response,
    }


def _apply_clahe_gray(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (4,4))
    ref_clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE).apply(blurred)
    return ref_clahe


def _compare_clahe_ssim(img1, img2, red_zones, threshold=0.55):
    results = []
    for rz in red_zones:
        x_start, y_start = rz['x'], rz['y']
        x_end = x_start + rz['width']
        y_end = y_start + rz['height']
        crop1 = img1[y_start:y_end, x_start:x_end]
        crop2 = img2[y_start:y_end, x_start:x_end]
        
        score, _ = ssim(crop1, crop2, full=True)
        results.append({
            **rz,
            "occluded": score < threshold,
            "occlusion_score": float(score),
            "mean_brightness": float(np.mean(crop2)),
            "original_mean_brightness": float(np.mean(crop1)),
        })
    return results