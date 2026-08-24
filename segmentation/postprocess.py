"""
segmentation/postprocess.py
Post-process the selected mask:
  - overlay on original image
  - extract region statistics for the LLM prompt
"""

import numpy as np
import cv2
from PIL import Image
from skimage import measure


# ── Overlay ──────────────────────────────────────────────────────────────────

def overlay_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,           # bool (H, W)
    # color: tuple = (255, 0, 0), # RGB red
    color: tuple = (0, 255, 100),  # bright green — visible on dark lesions
    alpha: float = 0.45,
) -> Image.Image:
    """
    Returns a PIL image with the mask overlaid as a semi-transparent color.
    Also draws the bounding box contour.
    """
    img_np = np.array(image.convert("RGB")).copy()
    H, W = img_np.shape[:2]

    # semi-transparent fill
    overlay = img_np.copy()
    overlay[mask] = (
        overlay[mask] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)

    # contour outline
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, thickness=2)

    return Image.fromarray(overlay)


# ── Region Statistics ─────────────────────────────────────────────────────────

def extract_region_stats(
    image: Image.Image,
    mask: np.ndarray,           # bool (H, W)
) -> dict:
    """
    Extract quantitative features from the segmented region.
    These become the structured input to the LLM for report generation.

    Returns dict with:
        area_pct        - % of image covered by mask
        location        - quadrant (top-left, top-right, bottom-left, bottom-right, center)
        mean_intensity  - mean grayscale intensity inside mask (0–255)
        std_intensity   - std of grayscale intensity inside mask
        contrast_ratio  - ratio of mean intensity inside vs outside mask
        irregularity    - perimeter² / (4π × area) — 1.0 = perfect circle, higher = irregular
        bbox_wh_ratio   - width/height of bounding box
        solidity        - area / convex hull area (1.0 = convex, <1 = concave/irregular)
    """
    img_gray = np.array(image.convert("L"))  # grayscale
    H, W = img_gray.shape
    total_pixels = H * W

    # ── area ────────────────────────────────────────────────────────────────
    area = int(mask.sum())
    area_pct = round(area / total_pixels * 100, 2)

    if area == 0:
        return {
            "area_pct": area_pct,
            "location": "not detected",
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "contrast_ratio": 0.0,
            "irregularity": 1.0,
            "bbox_wh_ratio": 0.0,
            "solidity": 0.0,
            "centroid": (0.0, 0.0),
            "image_size": (W, H),
        }

    # ── location (quadrant) ──────────────────────────────────────────────────
    ys, xs = np.where(mask)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    rel_x = centroid_x / W
    rel_y = centroid_y / H

    # check if close to center
    if 0.35 < rel_x < 0.65 and 0.35 < rel_y < 0.65:
        location = "center"
    elif rel_y < 0.5 and rel_x < 0.5:
        location = "upper-left"
    elif rel_y < 0.5 and rel_x >= 0.5:
        location = "upper-right"
    elif rel_y >= 0.5 and rel_x < 0.5:
        location = "lower-left"
    else:
        location = "lower-right"

    # ── intensity ────────────────────────────────────────────────────────────
    inside_pixels = img_gray[mask]
    outside_pixels = img_gray[~mask]

    mean_intensity = round(float(inside_pixels.mean()), 2)
    std_intensity = round(float(inside_pixels.std()), 2)

    outside_mean = float(outside_pixels.mean()) if outside_pixels.size > 0 else 1.0
    contrast_ratio = round(mean_intensity / (outside_mean + 1e-6), 3)

    # ── shape features ───────────────────────────────────────────────────────
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    irregularity = 1.0
    solidity = 1.0
    bbox_wh_ratio = 1.0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        contour_area = cv2.contourArea(largest)

        # irregularity: compactness (circle = 1.0, irregular = higher)
        if contour_area > 0:
            irregularity = round(
                (perimeter ** 2) / (4 * np.pi * contour_area), 3
            )

        # solidity
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = round(contour_area / hull_area, 3)

        # bbox aspect ratio
        x, y, w, h = cv2.boundingRect(largest)
        bbox_wh_ratio = round(w / (h + 1e-6), 3)

    return {
        "area_pct": area_pct,
        "location": location,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "contrast_ratio": contrast_ratio,
        "irregularity": irregularity,
        "bbox_wh_ratio": bbox_wh_ratio,
        "solidity": solidity,
        "centroid": (round(centroid_x, 1), round(centroid_y, 1)),
        "image_size": (W, H),
    }


def interpret_stats(stats: dict, scan_type: str) -> dict:
    """
    Convert raw stats into human-readable interpretations
    to pass to the LLM prompt alongside raw numbers.
    """
    interpretations = {}

    # size interpretation
    if stats["area_pct"] < 5:
        interpretations["size"] = "small"
    elif stats["area_pct"] < 20:
        interpretations["size"] = "moderate"
    else:
        interpretations["size"] = "large"

    # shape regularity
    if stats["irregularity"] < 1.5:
        interpretations["shape"] = "regular/round"
    elif stats["irregularity"] < 3.0:
        interpretations["shape"] = "mildly irregular"
    else:
        interpretations["shape"] = "highly irregular"

    # intensity (scan-type aware)
    if scan_type == "chest_xray":
        if stats["mean_intensity"] > 180:
            interpretations["intensity"] = "hyperdense/opaque"
        elif stats["mean_intensity"] < 80:
            interpretations["intensity"] = "hypodense/lucent"
        else:
            interpretations["intensity"] = "isodense to lung parenchyma"
    elif scan_type == "ultrasound":
        if stats["contrast_ratio"] < 0.8:
            interpretations["intensity"] = "hypoechoic relative to surrounding tissue"
        elif stats["contrast_ratio"] > 1.2:
            interpretations["intensity"] = "hyperechoic relative to surrounding tissue"
        else:
            interpretations["intensity"] = "isoechoic to surrounding tissue"
    else:  # skin_lesion
        if stats["contrast_ratio"] > 1.3:
            interpretations["intensity"] = "darker than surrounding skin"
        elif stats["contrast_ratio"] < 0.8:
            interpretations["intensity"] = "lighter than surrounding skin"
        else:
            interpretations["intensity"] = "similar intensity to surrounding skin"

    # solidity
    if stats["solidity"] > 0.90:
        interpretations["border"] = "well-defined convex border"
    elif stats["solidity"] > 0.75:
        interpretations["border"] = "mildly irregular border"
    else:
        interpretations["border"] = "irregular/concave border"

    return interpretations
