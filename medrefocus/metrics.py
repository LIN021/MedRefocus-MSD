"""Native-grid overlap and area-weighted surface metrics (0–1 internally)."""

from __future__ import annotations
import numpy as np
from surface_distance import metrics as surface


def dice_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    p, r = np.asarray(prediction, bool), np.asarray(reference, bool)
    total = int(p.sum()) + int(r.sum())
    return float(2 * (p & r).sum() / total) if total else float("nan")


def complete_miss(prediction: np.ndarray, reference: np.ndarray) -> int:
    p, r = np.asarray(prediction, bool), np.asarray(reference, bool)
    return int(r.any() and not (p & r).any())


def surface_metrics(prediction, reference, spacing, tolerance_mm=2.0):
    p, r = np.asarray(prediction, bool), np.asarray(reference, bool)
    if not p.any() and not r.any():
        return float("nan"), float("nan")
    if not p.any() or not r.any():
        return 0.0, float("inf")
    distances = surface.compute_surface_distances(r, p, tuple(float(v) for v in spacing))
    return (
        float(surface.compute_surface_dice_at_tolerance(distances, tolerance_mm)),
        float(surface.compute_robust_hausdorff(distances, 95)),
    )


def normalized_surface_dice(prediction, reference, spacing_zyx, tolerance_mm):
    return surface_metrics(prediction, reference, spacing_zyx, tolerance_mm)[0]


def acceptance_label(base, proposal, reference, spacing, config):
    """Reference-dependent supervision, used only by training/evaluation."""
    base, proposal, reference = base == 2, proposal == 2, reference == 2
    delta = dice_score(proposal, reference) - dice_score(base, reference)
    nsd_delta = normalized_surface_dice(
        proposal, reference, spacing, 2.0
    ) - normalized_surface_dice(base, reference, spacing, 2.0)
    recovered = complete_miss(base, reference) and not complete_miss(proposal, reference)
    label = int(
        recovered
        or (delta > config["minimum_dice_gain"] and nsd_delta >= config["minimum_nsd_gain"])
    )
    weight = (
        config["recovered_miss_weight"]
        if recovered
        else config["severe_harm_weight"]
        if delta <= -0.05
        else config["moderate_harm_weight"]
        if delta <= -0.005
        else config["default_weight"]
    )
    return label, float(weight)
