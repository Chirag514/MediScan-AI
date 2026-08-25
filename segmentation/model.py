"""
segmentation/model.py
ONNX Runtime inference for SegFormer-b2 fine-tuned models.
Replaces SAM2 — deterministic, CPU-fast, no mask selection needed.

Models (place in ./models/):
    segformer_isic.onnx  — skin lesion segmentation
    segformer_busi.onnx  — breast ultrasound segmentation
    segformer_xray.onnx  — lung segmentation (chest X-ray)
"""

import numpy as np
from PIL import Image
import onnxruntime as ort
import torch.nn.functional as F
import torch
from pathlib import Path

torch.set_num_threads(1)


# ── Model registry ───────────────────────────────────────────────────────────
MODELS = {
    "skin_lesion": "models/segformer_isic.onnx",
    "ultrasound":  "models/segformer_busi.onnx",
    "chest_xray":  "models/segformer_xray.onnx",
}
REPO_ROOT = Path(__file__).resolve().parent.parent

# ImageNet normalization (same as training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 512  # must match training img_size

# ── Session cache — load once per scan type ───────────────────────────────────
_sessions: dict = {}


def load_session(scan_type: str) -> ort.InferenceSession:
    global _sessions
    if scan_type not in _sessions:
        model_path = MODELS.get(scan_type)
        if model_path is None:
            raise ValueError(f"Unknown scan type: {scan_type}. "
                             f"Choose from {list(MODELS.keys())}")
        model_path = REPO_ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                f"Make sure you placed the .onnx files in the models/ folder."
            )
        print(f"Loading {scan_type} model from {model_path}...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1   # use 1 CPU thread
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        _sessions[scan_type] = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        print(f"  {scan_type} model loaded.")
    return _sessions[scan_type]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize to 512×512, normalize with ImageNet stats.
    Returns float32 array of shape (1, 3, 512, 512).
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = (arr - MEAN) / STD                                # normalize
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]           # (1, 3, H, W)
    return arr.astype(np.float32)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_segmentation(
    image: Image.Image,
    scan_type: str,
) -> tuple[np.ndarray, float]:
    """
    Run ONNX segmentation model on a PIL image.

    Returns:
        mask  : bool np.ndarray of shape (H, W) — original image size
        score : float confidence score (mean foreground probability)
    """
    session   = load_session(scan_type)
    orig_w, orig_h = image.size

    # preprocess
    input_arr = preprocess(image)                           # (1, 3, 512, 512)

    # run ONNX inference
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_arr})[0]  # (1, 2, H/4, W/4)

    # upsample logits to 512×512 then to original size
    logits_t  = torch.from_numpy(logits)                    # (1, 2, h, w)
    logits_up = F.interpolate(
        logits_t,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )                                                       # (1, 2, 512, 512)

    # foreground channel (index 1) → probability
    prob = torch.sigmoid(logits_up[0, 1]).numpy()           # (512, 512) float

    # threshold → binary mask at 512×512
    mask_512 = (prob > 0.5).astype(np.uint8)

    # resize back to original image size
    mask_img  = Image.fromarray(mask_512 * 255).resize(
        (orig_w, orig_h), Image.NEAREST
    )
    mask_orig = np.array(mask_img) > 127                    # bool (H, W)

    # confidence = mean probability in foreground region
    fg_pixels = prob[mask_512 == 1]
    score     = float(fg_pixels.mean()) if len(fg_pixels) > 0 else 0.0

    return mask_orig, score


# ── Compatibility wrapper (keeps app.py interface unchanged) ──────────────────

def generate_masks(image: Image.Image, scan_type: str = "skin_lesion"):
    """
    Drop-in replacement for the old SAM2 generate_masks().
    Returns a list with a single mask dict (same format as before).
    """
    mask, score = run_segmentation(image, scan_type)
    area        = int(mask.sum())

    ys, xs = np.where(mask)
    if len(xs) > 0:
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min()), int(ys.max() - ys.min())]
    else:
        bbox = [0, 0, image.size[0], image.size[1]]

    return [{
        "segmentation": mask,
        "area":         area,
        "bbox":         bbox,
        "score":        score,
    }]


def select_best_mask(masks: list[dict], image_size: tuple) -> dict | None:
    """
    With ONNX models there's always exactly one mask — just return it.
    Kept for compatibility with app.py.
    """
    if not masks:
        return None
    return masks[0]


def get_all_mask_stats(masks: list[dict], image_size: tuple) -> list[dict]:
    """
    Returns stats for the single ONNX mask.
    Kept for compatibility with app.py debug expander.
    """
    H, W      = image_size
    total_px  = H * W
    m         = masks[0]
    area_pct  = m["area"] / total_px * 100

    seg       = m["segmentation"]
    ys, xs    = np.where(seg)
    cx        = float(xs.mean()) if len(xs) > 0 else 0
    cy        = float(ys.mean()) if len(ys) > 0 else 0

    return [{
        "rank":      0,
        "area_pct":  round(area_pct, 2),
        "score":     round(m["score"], 3),
        "centroid":  (round(cx, 1), round(cy, 1)),
        "model":     "SegFormer-b2 (fine-tuned)",
    }]
