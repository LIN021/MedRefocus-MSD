#!/usr/bin/env python3
"""Unified evaluation for 3D medical image segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure, label as connected_components
from scipy.optimize import linear_sum_assignment

try:
    from surface_distance import metrics as surface_metrics
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: surface-distance. Install it with:\n"
        "python -m pip install surface-distance"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate matching 3D segmentation files with one fixed metric protocol."
    )
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--labels", required=True, nargs="+", type=int)
    parser.add_argument("--label-names", required=True, nargs="+")
    parser.add_argument(
        "--nsd-tolerance-mm",
        required=True,
        nargs="+",
        type=float,
        help="One value for all labels, or one value per label.",
    )
    parser.add_argument("--file-ending", default=".nii.gz")
    parser.add_argument(
        "--allow-prediction-subset",
        action="store_true",
        help=(
            "Evaluate only prediction files that have matching ground truth. "
            "Use this for a single validation fold whose GT directory contains all cases."
        ),
    )
    parser.add_argument(
        "--lesion-connectivity",
        choices=(1, 2, 3),
        default=3,
        type=int,
        help="3D connected-component connectivity: 1=6, 2=18, 3=26.",
    )
    parser.add_argument(
        "--lesion-min-volume-mm3",
        default=0.0,
        type=float,
        help="Ignore connected components smaller than this volume.",
    )
    return parser.parse_args()


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_or_none(v) for v in value]
    return value


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def verify_geometry(gt: sitk.Image, pred: sitk.Image, case: str) -> None:
    checks = {
        "size": np.array_equal(gt.GetSize(), pred.GetSize()),
        "spacing": np.allclose(gt.GetSpacing(), pred.GetSpacing(), atol=1e-5),
        "origin": np.allclose(gt.GetOrigin(), pred.GetOrigin(), atol=1e-4),
        "direction": np.allclose(gt.GetDirection(), pred.GetDirection(), atol=1e-5),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{case}: geometry mismatch in {', '.join(failed)}")


def overlap_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    gt_count = tp + fn
    pred_count = tp + fp

    if gt_count == 0 and pred_count == 0:
        dice = iou = precision = recall = float("nan")
    else:
        dice = safe_div(2 * tp, 2 * tp + fp + fn)
        iou = safe_div(tp, tp + fp + fn)
        precision = safe_div(tp, pred_count) if pred_count else 0.0
        recall = safe_div(tp, gt_count) if gt_count else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_voxels": gt_count,
        "pred_voxels": pred_count,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def surface_measures(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    tolerance_mm: float,
) -> tuple[float, float, float]:
    if not gt.any() and not pred.any():
        return float("nan"), float("nan"), float("nan")
    if not gt.any() or not pred.any():
        return float("inf"), 0.0, float("inf")

    distances = surface_metrics.compute_surface_distances(gt, pred, spacing_zyx)
    hd95 = float(surface_metrics.compute_robust_hausdorff(distances, 95))
    nsd = float(surface_metrics.compute_surface_dice_at_tolerance(distances, tolerance_mm))
    assd_pair = surface_metrics.compute_average_surface_distance(distances)
    assd = float(np.mean(assd_pair))
    return hd95, nsd, assd


def filtered_components(
    mask: np.ndarray,
    voxel_volume_mm3: float,
    connectivity: int,
    min_volume_mm3: float,
) -> tuple[np.ndarray, list[int]]:
    structure = generate_binary_structure(3, connectivity)
    component_map, count = connected_components(mask, structure=structure)
    sizes = np.bincount(component_map.ravel(), minlength=count + 1)

    kept_old_ids = [
        component_id
        for component_id in range(1, count + 1)
        if sizes[component_id] * voxel_volume_mm3 >= min_volume_mm3
    ]

    filtered = np.zeros_like(component_map, dtype=np.int32)
    kept_sizes: list[int] = []
    for new_id, old_id in enumerate(kept_old_ids, start=1):
        filtered[component_map == old_id] = new_id
        kept_sizes.append(int(sizes[old_id]))
    return filtered, kept_sizes


def lesion_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    voxel_volume_mm3: float,
    connectivity: int,
    min_volume_mm3: float,
) -> dict[str, float | int]:
    gt_cc, gt_sizes = filtered_components(gt, voxel_volume_mm3, connectivity, min_volume_mm3)
    pred_cc, pred_sizes = filtered_components(pred, voxel_volume_mm3, connectivity, min_volume_mm3)
    n_gt = len(gt_sizes)
    n_pred = len(pred_sizes)

    if n_gt == 0 and n_pred == 0:
        return {
            "gt_lesions": 0,
            "pred_lesions": 0,
            "matched_lesions": 0,
            "lesion_precision": float("nan"),
            "lesion_recall": float("nan"),
            "lesion_f1": float("nan"),
            "matched_lesion_mean_dice": float("nan"),
        }

    intersections = np.zeros((n_gt, n_pred), dtype=np.int64)
    overlap = np.logical_and(gt_cc > 0, pred_cc > 0)
    if overlap.any():
        pairs = np.stack((gt_cc[overlap] - 1, pred_cc[overlap] - 1), axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        intersections[unique_pairs[:, 0], unique_pairs[:, 1]] = counts

    if n_gt and n_pred:
        denominator = (
            np.asarray(gt_sizes, dtype=np.float64)[:, None]
            + np.asarray(pred_sizes, dtype=np.float64)[None, :]
        )
        pair_dice = 2.0 * intersections / denominator
        rows, cols = linear_sum_assignment(-pair_dice)
        valid = intersections[rows, cols] > 0
        matched_scores = pair_dice[rows[valid], cols[valid]]
    else:
        matched_scores = np.asarray([], dtype=np.float64)

    matched = int(len(matched_scores))
    precision = safe_div(matched, n_pred)
    recall = safe_div(matched, n_gt)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "gt_lesions": n_gt,
        "pred_lesions": n_pred,
        "matched_lesions": matched,
        "lesion_precision": precision,
        "lesion_recall": recall,
        "lesion_f1": f1,
        "matched_lesion_mean_dice": (float(matched_scores.mean()) if matched else 0.0),
    }


def evaluate_case(
    case: str,
    gt_path: Path,
    pred_path: Path,
    labels: list[int],
    label_names: list[str],
    tolerances: list[float],
    connectivity: int,
    lesion_min_volume_mm3: float,
) -> list[dict[str, Any]]:
    gt_image = sitk.ReadImage(str(gt_path))
    pred_image = sitk.ReadImage(str(pred_path))
    verify_geometry(gt_image, pred_image, case)

    gt_array = sitk.GetArrayFromImage(gt_image)
    pred_array = sitk.GetArrayFromImage(pred_image)
    spacing_xyz = tuple(float(v) for v in gt_image.GetSpacing())
    spacing_zyx = spacing_xyz[::-1]
    voxel_volume_mm3 = float(np.prod(spacing_xyz))

    rows = []
    for label_value, label_name, tolerance in zip(labels, label_names, tolerances):
        gt_mask = gt_array == label_value
        pred_mask = pred_array == label_value
        overlap = overlap_metrics(gt_mask, pred_mask)
        hd95, nsd, assd = surface_measures(gt_mask, pred_mask, spacing_zyx, tolerance)
        lesions = lesion_metrics(
            gt_mask,
            pred_mask,
            voxel_volume_mm3,
            connectivity,
            lesion_min_volume_mm3,
        )

        gt_volume = overlap["gt_voxels"] * voxel_volume_mm3
        pred_volume = overlap["pred_voxels"] * voxel_volume_mm3
        volume_error = pred_volume - gt_volume
        rows.append(
            {
                "case": case,
                "label": label_value,
                "label_name": label_name,
                "spacing_x": spacing_xyz[0],
                "spacing_y": spacing_xyz[1],
                "spacing_z": spacing_xyz[2],
                **overlap,
                "gt_volume_mm3": gt_volume,
                "pred_volume_mm3": pred_volume,
                "volume_error_mm3": volume_error,
                "absolute_volume_error_mm3": abs(volume_error),
                "relative_volume_error": safe_div(volume_error, gt_volume),
                "hd95_mm": hd95,
                "assd_mm": assd,
                "nsd_tolerance_mm": tolerance,
                "nsd": nsd,
                "complete_miss": int(overlap["gt_voxels"] > 0 and overlap["tp"] == 0),
                "false_positive_case": int(
                    overlap["gt_voxels"] == 0 and overlap["pred_voxels"] > 0
                ),
                **lesions,
            }
        )
    return rows


def nan_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "q25": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "q75": float(np.percentile(finite, 75)) if finite.size else float("nan"),
        "valid_cases": int(finite.size),
        "infinite_cases": int(np.isinf(array).sum()),
        "nan_cases": int(np.isnan(array).sum()),
    }


def summarize(rows: list[dict[str, Any]], label_names: list[str]) -> list[dict[str, Any]]:
    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm",
        "assd_mm",
        "nsd",
        "absolute_volume_error_mm3",
        "lesion_precision",
        "lesion_recall",
        "lesion_f1",
        "matched_lesion_mean_dice",
    ]
    summaries = []
    for label_name in label_names:
        subset = [row for row in rows if row["label_name"] == label_name]
        record: dict[str, Any] = {
            "label_name": label_name,
            "cases": len(subset),
            "complete_misses": sum(row["complete_miss"] for row in subset),
            "false_positive_cases": sum(row["false_positive_case"] for row in subset),
            "gt_lesions": sum(row["gt_lesions"] for row in subset),
            "pred_lesions": sum(row["pred_lesions"] for row in subset),
            "matched_lesions": sum(row["matched_lesions"] for row in subset),
        }
        for metric_name in metric_names:
            stats = nan_stats([float(row[metric_name]) for row in subset])
            for stat_name, value in stats.items():
                record[f"{metric_name}_{stat_name}"] = value
        summaries.append(record)
    return summaries


def volume_stratification(
    rows: list[dict[str, Any]], label_names: list[str]
) -> list[dict[str, Any]]:
    output = []
    for label_name in label_names:
        subset = [
            row for row in rows if row["label_name"] == label_name and row["gt_volume_mm3"] > 0
        ]
        if not subset:
            continue
        volumes = np.asarray([row["gt_volume_mm3"] for row in subset])
        q1, q2 = np.quantile(volumes, [1 / 3, 2 / 3])
        for group_name, lower, upper in [
            ("small", -np.inf, q1),
            ("medium", q1, q2),
            ("large", q2, np.inf),
        ]:
            group = [row for row in subset if lower < row["gt_volume_mm3"] <= upper]
            if not group:
                continue
            output.append(
                {
                    "label_name": label_name,
                    "volume_group": group_name,
                    "lower_mm3_exclusive": lower,
                    "upper_mm3_inclusive": upper,
                    "cases": len(group),
                    "mean_dice": nan_stats([row["dice"] for row in group])["mean"],
                    "median_dice": nan_stats([row["dice"] for row in group])["median"],
                    "mean_nsd": nan_stats([row["nsd"] for row in group])["mean"],
                    "complete_misses": sum(row["complete_miss"] for row in group),
                    "median_gt_volume_mm3": float(
                        np.median([row["gt_volume_mm3"] for row in group])
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(finite_or_none(row))


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.label_names):
        raise ValueError("--labels and --label-names must have equal length")
    if len(args.nsd_tolerance_mm) == 1:
        tolerances = args.nsd_tolerance_mm * len(args.labels)
    elif len(args.nsd_tolerance_mm) == len(args.labels):
        tolerances = args.nsd_tolerance_mm
    else:
        raise ValueError("--nsd-tolerance-mm requires one value or one value per label")

    all_gt_files = sorted(args.gt_dir.glob(f"*{args.file_ending}"))
    pred_files = sorted(args.pred_dir.glob(f"*{args.file_ending}"))
    if not all_gt_files:
        raise FileNotFoundError(f"No {args.file_ending} files in {args.gt_dir}")
    if not pred_files:
        raise FileNotFoundError(f"No {args.file_ending} files in {args.pred_dir}")

    missing_predictions = [
        gt_path.name for gt_path in all_gt_files if not (args.pred_dir / gt_path.name).is_file()
    ]
    extra_predictions = sorted(
        path.name for path in pred_files if not (args.gt_dir / path.name).is_file()
    )
    if extra_predictions:
        raise FileNotFoundError(
            f"Predictions without matching ground truth: {extra_predictions[:5]}"
        )
    if missing_predictions and not args.allow_prediction_subset:
        raise FileNotFoundError(
            f"Missing {len(missing_predictions)} predictions, first: "
            f"{missing_predictions[:5]}. If this is a validation subset, add "
            "--allow-prediction-subset."
        )
    if args.allow_prediction_subset:
        gt_files = [args.gt_dir / pred_path.name for pred_path in pred_files]
    else:
        gt_files = all_gt_files

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, gt_path in enumerate(gt_files, start=1):
        case = gt_path.name[: -len(args.file_ending)]
        print(f"[{index}/{len(gt_files)}] {case}", flush=True)
        rows.extend(
            evaluate_case(
                case,
                gt_path,
                args.pred_dir / gt_path.name,
                args.labels,
                args.label_names,
                tolerances,
                args.lesion_connectivity,
                args.lesion_min_volume_mm3,
            )
        )

    class_summary = summarize(rows, args.label_names)
    volume_summary = volume_stratification(rows, args.label_names)
    failures = [
        row
        for row in rows
        if row["complete_miss"] or row["false_positive_case"] or row["dice"] < 0.2
    ]
    lesion_rows = [
        {
            key: row[key]
            for key in [
                "case",
                "label",
                "label_name",
                "gt_lesions",
                "pred_lesions",
                "matched_lesions",
                "lesion_precision",
                "lesion_recall",
                "lesion_f1",
                "matched_lesion_mean_dice",
            ]
        }
        for row in rows
    ]

    write_csv(args.output_dir / "case_metrics.csv", rows)
    write_csv(args.output_dir / "class_summary.csv", class_summary)
    write_csv(args.output_dir / "lesion_metrics.csv", lesion_rows)
    write_csv(args.output_dir / "volume_stratified.csv", volume_summary)
    write_csv(args.output_dir / "failures.csv", failures)

    report = {
        "protocol": {
            "labels": dict(zip(args.label_names, args.labels)),
            "nsd_tolerance_mm": dict(zip(args.label_names, tolerances)),
            "lesion_connectivity": args.lesion_connectivity,
            "lesion_min_volume_mm3": args.lesion_min_volume_mm3,
            "empty_gt_and_pred_policy": "overlap and surface metrics are excluded (NaN)",
            "one_empty_mask_policy": "Dice/IoU=0, NSD=0, HD95/ASSD=inf",
        },
        "inputs": {
            "gt_dir": str(args.gt_dir.resolve()),
            "pred_dir": str(args.pred_dir.resolve()),
            "cases": len(gt_files),
            "prediction_subset_enabled": args.allow_prediction_subset,
            "ground_truth_cases_not_evaluated": (
                missing_predictions if args.allow_prediction_subset else []
            ),
        },
        "class_summary": class_summary,
        "volume_stratified": volume_summary,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(finite_or_none(report), handle, ensure_ascii=False, indent=2)

    print("\nEvaluation complete")
    for record in class_summary:
        print(
            f"{record['label_name']}: "
            f"Dice={record['dice_mean']:.4f}, "
            f"NSD={record['nsd_mean']:.4f}, "
            f"HD95={record['hd95_mm_mean']:.2f} mm, "
            f"misses={record['complete_misses']}"
        )
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
