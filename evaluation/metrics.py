"""
evaluation/metrics.py
Dice Score and IoU for evaluating segmentation quality.
Use these when you have ground truth masks (e.g., ISIC dataset).
"""

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice Similarity Coefficient.
    Both pred and target should be boolean/binary arrays of same shape.
    Returns value in [0, 1]. 1.0 = perfect match.
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = (pred & target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Intersection over Union (Jaccard Index).
    Returns value in [0, 1]. 1.0 = perfect match.
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return float((intersection + smooth) / (union + smooth))


def evaluate_dataset(predictions: list, targets: list) -> dict:
    """
    Evaluate over a list of (pred_mask, target_mask) pairs.
    Returns mean Dice, mean IoU, and per-sample scores.
    """
    if not predictions or not targets:
        raise ValueError("predictions and targets must not be empty")
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")

    dice_scores = []
    iou_scores = []

    for pred, target in zip(predictions, targets):
        dice_scores.append(dice_score(pred, target))
        iou_scores.append(iou_score(pred, target))

    return {
        "mean_dice": round(float(np.mean(dice_scores)), 4),
        "std_dice": round(float(np.std(dice_scores)), 4),
        "mean_iou": round(float(np.mean(iou_scores)), 4),
        "std_iou": round(float(np.std(iou_scores)), 4),
        "per_sample_dice": dice_scores,
        "per_sample_iou": iou_scores,
    }
