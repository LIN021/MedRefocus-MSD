from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import ndimage

from medrefocus.pcsc.feature_names import FEATURE_NAMES


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def component_stats(mask: np.ndarray) -> dict[str, int]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "count": int(count),
        "largest": int(sizes.max()) if sizes.size else 0,
        "small_lt10": int((sizes < 10).sum()),
        "small_lt50": int((sizes < 50).sum()),
    }


def _planner_value(row: pd.Series, name: str, fallback: str | None = None) -> float:
    value = row.get(name, np.nan)
    if pd.isna(value) and fallback is not None:
        value = row.get(fallback, np.nan)
    return float(value) if not pd.isna(value) else np.nan


def extract_features(
    row: pd.Series,
    base: np.ndarray,
    proposal: np.ndarray,
    local_probability: np.ndarray,
    spacing: np.ndarray,
    target_label: int,
) -> dict[str, float]:
    if base.shape != proposal.shape:
        raise ValueError("Base and proposal shapes differ")
    z0, z1, y0, y1, x0, x1 = (int(row[key]) for key in ("z0", "z1", "y0", "y1", "x0", "x1"))
    box = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    base_roi = base[box]
    proposal_roi = proposal[box]
    if local_probability.shape != base_roi.shape:
        raise ValueError(
            f"Local probability shape {local_probability.shape} != ROI {base_roi.shape}"
        )
    base_target = base_roi == target_label
    proposal_target = proposal_roi == target_label
    added = proposal_target & ~base_target
    intersection = proposal_target & base_target
    union = proposal_target | base_target
    base_components = component_stats(base_target)
    proposal_components = component_stats(proposal_target)
    added_components = component_stats(added)
    roi_voxels = int(base_roi.size)
    base_voxels = int(base_target.sum())
    proposal_voxels = int(proposal_target.sum())
    added_voxels = int(added.sum())
    intersection_voxels = int(intersection.sum())
    union_voxels = int(union.sum())
    voxel_volume = float(np.prod(spacing))
    base_volume = float((base == target_label).sum() * voxel_volume)
    proposal_volume = float((proposal == target_label).sum() * voxel_volume)
    outside_foreground = int((added & (base_roi == 0)).sum())

    if added_voxels and base_voxels:
        distances = ndimage.distance_transform_edt(~base_target)[added]
        distance_values = (
            float(distances.min()),
            float(distances.mean()),
            float(np.percentile(distances, 90)),
        )
    elif added_voxels:
        distance_values = (999.0, 999.0, 999.0)
    else:
        distance_values = (0.0, 0.0, 0.0)

    result: dict[str, float] = {
        "abs_delta_pred_volume_mm3": abs(proposal_volume - base_volume),
        "added_component_count": added_components["count"],
        "added_largest_component_vox": added_components["largest"],
        "added_outside_base_foreground_frac": safe_ratio(outside_foreground, added_voxels),
        "added_outside_base_foreground_vox": outside_foreground,
        "added_small_component_count_lt10": added_components["small_lt10"],
        "added_small_component_count_lt50": added_components["small_lt50"],
        "added_to_base_tumor_mean_dist_vox": distance_values[1],
        "added_to_base_tumor_min_dist_vox": distance_values[0],
        "added_to_base_tumor_p90_dist_vox": distance_values[2],
        "base_pred_volume_mm3": base_volume,
        "base_roi_component_count": base_components["count"],
        "base_roi_largest_component_vox": base_components["largest"],
        "center_x": _planner_value(row, "center_x", None),
        "center_y": _planner_value(row, "center_y", None),
        "center_z": _planner_value(row, "center_z", None),
        "delta_pred_volume_mm3": proposal_volume - base_volume,
        "planner_center_x_norm": _planner_value(row, "center_x_norm"),
        "planner_center_y_norm": _planner_value(row, "center_y_norm"),
        "planner_center_z_norm": _planner_value(row, "center_z_norm"),
        "planner_ucrp_score": _planner_value(row, "ucrp_score"),
        "planner_rank": _planner_value(row, "rank"),
        "planner_roi_volume_ratio": _planner_value(row, "roi_volume_ratio"),
        "planner_score_value": _planner_value(row, "score_value"),
        "pred_volume_ratio": safe_ratio(proposal_volume + 1.0, base_volume + 1.0),
        "roi_added_over_proposal": safe_ratio(added_voxels, proposal_voxels),
        "roi_added_tumor_frac": safe_ratio(added_voxels, roi_voxels),
        "roi_added_tumor_vox": added_voxels,
        "roi_base_tumor_frac": safe_ratio(base_voxels, roi_voxels),
        "roi_base_tumor_vox": base_voxels,
        "roi_intersection_tumor_vox": intersection_voxels,
        "roi_pred_jaccard_agreement": safe_ratio(intersection_voxels, union_voxels),
        "roi_score": _planner_value(row, "ucrp_score"),
        "roi_union_tumor_vox": union_voxels,
        "roi_proposal_over_base_tumor_ratio": safe_ratio(proposal_voxels + 1, base_voxels + 1),
        "roi_proposal_tumor_frac": safe_ratio(proposal_voxels, roi_voxels),
        "roi_proposal_tumor_vox": proposal_voxels,
        "roi_volume_fraction": safe_ratio(roi_voxels, base.size),
        "roi_vox_x": x1 - x0,
        "roi_vox_y": y1 - y0,
        "roi_vox_z": z1 - z0,
        "shape_z": base.shape[0],
        "spacing_x": float(spacing[2]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[0]),
        "proposal_pred_volume_mm3": proposal_volume,
        "proposal_roi_component_count": proposal_components["count"],
        "proposal_roi_largest_component_vox": proposal_components["largest"],
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "z0": z0,
        "z1": z1,
    }
    for source in (
        "full_entropy_max",
        "full_entropy_mean",
        "full_entropy_p90",
        "full_entropy_sum",
        "full_tumor_max",
        "full_tumor_mean",
        "full_tumor_p90",
        "full_tumor_sum",
        "obs_entropy_max",
        "obs_entropy_mean",
        "obs_entropy_p90",
        "obs_entropy_sum",
        "obs_pancreas_max",
        "obs_pancreas_mean",
        "obs_pancreas_p90",
        "obs_pancreas_sum",
        "obs_tumor_max",
        "obs_tumor_mean",
        "obs_tumor_p90",
        "obs_tumor_sum",
    ):
        result[f"planner_{source}"] = _planner_value(row, source)

    threshold = float(row["local_threshold"])
    current = base_target
    local_mask = local_probability >= threshold
    local_added = local_mask & ~current
    local_overlap = local_mask & current
    local_outside = ~local_mask
    probability = np.asarray(local_probability, dtype=np.float64)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    local_entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    local_components = component_stats(local_mask)
    local_added_components = component_stats(local_added)
    local_mask_voxels = int(local_mask.sum())
    local_added_voxels = int(local_added.sum())
    local_overlap_voxels = int(local_overlap.sum())
    outside_mean = safe_mean(probability[local_outside])
    result.update(
        {
            "local_threshold": threshold,
            "local_has_current_tumor": float(row["local_has_current_tumor"]),
            "local_applied": float(row["local_applied"]),
            "local_prob_mean": float(probability.mean()),
            "local_prob_std": float(probability.std()),
            "local_prob_max": float(probability.max()),
            "local_prob_p50": float(np.percentile(probability, 50)),
            "local_prob_p75": float(np.percentile(probability, 75)),
            "local_prob_p90": float(np.percentile(probability, 90)),
            "local_prob_p95": float(np.percentile(probability, 95)),
            "local_prob_p99": float(np.percentile(probability, 99)),
            "local_entropy_mean": float(local_entropy.mean()),
            "local_entropy_std": float(local_entropy.std()),
            "local_entropy_p90": float(np.percentile(local_entropy, 90)),
            "local_entropy_p95": float(np.percentile(local_entropy, 95)),
            "local_mask_component_count": local_components["count"],
            "local_mask_largest_component_vox": local_components["largest"],
            "local_mask_small_component_count_lt10": local_components["small_lt10"],
            "local_mask_small_component_count_lt50": local_components["small_lt50"],
            "local_added_component_count": local_added_components["count"],
            "local_added_largest_component_vox": local_added_components["largest"],
            "local_added_small_component_count_lt10": local_added_components["small_lt10"],
            "local_added_small_component_count_lt50": local_added_components["small_lt50"],
            "local_prob_on_current_tumor_mean": safe_mean(probability[current]),
            "local_prob_outside_current_tumor_mean": safe_mean(probability[~current]),
            "local_prob_on_added_mean": safe_mean(probability[local_added]),
            "local_prob_on_overlap_mean": safe_mean(probability[local_overlap]),
            "local_prob_outside_local_mask_mean": outside_mean,
            "local_current_tumor_vox": int(current.sum()),
            "local_mask_tumor_vox": local_mask_voxels,
            "local_added_vox": local_added_voxels,
            "local_overlap_vox": local_overlap_voxels,
            "local_added_fraction_of_mask": safe_ratio(local_added_voxels, local_mask_voxels),
            "local_overlap_fraction_of_mask": safe_ratio(local_overlap_voxels, local_mask_voxels),
            "local_mask_current_jaccard": safe_ratio(
                int((local_mask & current).sum()), int((local_mask | current).sum())
            ),
            "local_prob_added_vs_outside_contrast": safe_mean(probability[local_added])
            - outside_mean,
            "local_prob_overlap_vs_outside_contrast": safe_mean(probability[local_overlap])
            - outside_mean,
        }
    )
    for threshold_value in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tag = f"{threshold_value:.1f}".replace(".", "p")
        result[f"local_frac_ge_{tag}"] = float((probability >= threshold_value).mean())

    missing = set(FEATURE_NAMES) - set(result)
    extra = set(result) - set(FEATURE_NAMES)
    if missing or extra:
        raise RuntimeError(f"PCSC feature-contract mismatch: missing={missing}, extra={extra}")
    return {name: float(result[name]) for name in FEATURE_NAMES}
