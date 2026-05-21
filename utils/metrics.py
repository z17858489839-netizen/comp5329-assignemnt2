"""
utils/metrics.py
VQA accuracy, token-reduction metrics, and per-level statistics.
"""

import numpy as np
from collections import Counter


# ── Token count per resolution level ─────────────────────────────────────────
# Approximate values for Qwen2.5-VL with 448px base and 14px patches.
# Actual count depends on image aspect ratio; these are representative.
TOKENS_PER_LEVEL = {0: 64, 1: 256, 2: 1024}   # low / mid / full


def token_reduction_pct(predicted_levels: list) -> float:
    """Mean token reduction % vs always using full resolution."""
    full_tokens = TOKENS_PER_LEVEL[2]
    used = np.mean([TOKENS_PER_LEVEL[k] for k in predicted_levels])
    return round((1 - used / full_tokens) * 100, 1)


def avg_tokens(predicted_levels: list) -> float:
    return round(float(np.mean([TOKENS_PER_LEVEL[k] for k in predicted_levels])), 1)


def level_distribution(predicted_levels: list) -> dict:
    total = len(predicted_levels)
    counts = Counter(predicted_levels)
    return {
        "low_pct":  round(counts.get(0, 0) / total * 100, 1),
        "mid_pct":  round(counts.get(1, 0) / total * 100, 1),
        "full_pct": round(counts.get(2, 0) / total * 100, 1),
    }


def vqa_accuracy(predictions: list, gt_answers: list) -> float:
    """
    Standard VQA accuracy: prediction matches any of the GT answers.
    predictions  : list of str
    gt_answers   : list of list[str]
    """
    from utils.vlm_inference import is_correct
    correct = sum(is_correct(p, g) for p, g in zip(predictions, gt_answers))
    return round(correct / len(predictions) * 100, 2)


def pseudo_label_accuracy(predicted_levels: list, true_labels: list) -> float:
    """How accurately QVFP predicts the pseudo label (classifier accuracy)."""
    correct = sum(p == t for p, t in zip(predicted_levels, true_labels))
    return round(correct / len(predicted_levels) * 100, 2)


def summarise_results(
    method_name: str,
    vqa_acc: float,
    predicted_levels: list,
) -> dict:
    dist = level_distribution(predicted_levels)
    return {
        "method":           method_name,
        "vqa_accuracy":     vqa_acc,
        "avg_tokens":       avg_tokens(predicted_levels),
        "token_reduction%": token_reduction_pct(predicted_levels),
        **dist,
    }
