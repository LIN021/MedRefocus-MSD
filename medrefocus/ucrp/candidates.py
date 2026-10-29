from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from medrefocus.common import read_splits, validation_fold_map
from medrefocus.metrics import dice_score
from medrefocus.io import (
    probability_path,
    read_nifti,
    read_probability_npz,
    segmentation_path,
    spacing_zyx,
)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    logs = np.log(np.clip(probabilities, np.finfo(np.float32).tiny, 1.0))
    return -np.sum(probabilities * logs, axis=0)


def box_from_center(
    center: np.ndarray, extent: np.ndarray, shape: np.ndarray
) -> tuple[int, int, int, int, int, int]:
    starts: list[int] = []
    ends: list[int] = []
    for coordinate, size, limit in zip(center, extent, shape):
        size = int(min(size, limit))
        start = int(round(float(coordinate) - size / 2))
        start = max(0, min(start, int(limit) - size))
        starts.append(start)
        ends.append(start + size)
    return starts[0], ends[0], starts[1], ends[1], starts[2], ends[2]


def top_local_maxima(
    score: np.ndarray, count: int, suppression_radius: np.ndarray, valid: np.ndarray
) -> list[tuple[np.ndarray, float]]:
    work = np.asarray(score, dtype=np.float32).copy()
    work[~valid] = -np.inf
    selected: list[tuple[np.ndarray, float]] = []
    for _ in range(count):
        flat_index = int(np.argmax(work))
        value = float(work.flat[flat_index])
        if not np.isfinite(value) or value <= 0:
            break
        center = np.asarray(np.unravel_index(flat_index, work.shape), dtype=int)
        selected.append((center, value))
        slices = tuple(
            slice(max(0, int(c - radius)), min(limit, int(c + radius + 1)))
            for c, radius, limit in zip(center, suppression_radius, work.shape)
        )
        work[slices] = -np.inf
    return selected


def _case_rng(seed: int, case: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{case}".encode()).digest()
    derived = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived)


def uniform_centers(
    valid: np.ndarray, count: int, generator: np.random.Generator
) -> list[tuple[np.ndarray, float]]:
    coordinates = np.argwhere(valid)
    if not len(coordinates):
        return []
    indices = generator.choice(len(coordinates), size=min(count, len(coordinates)), replace=False)
    return [(coordinates[index].astype(int), 0.0) for index in indices]


def region_stats(array: np.ndarray, box: tuple[int, ...]) -> dict[str, float]:
    z0, z1, y0, y1, x0, x1 = box
    values = array[z0:z1, y0:y1, x0:x1]
    return {
        "mean": float(values.mean()),
        "max": float(values.max()),
        "p90": float(np.percentile(values, 90)),
        "sum": float(values.sum()),
    }


def build_case_candidates(
    case: str,
    fold: int,
    primary_dir: Path,
    auxiliary_dir: Path,
    reference_dir: Path | None,
    roi_mm: np.ndarray,
    maxima_per_map: int,
    uniform_count: int,
    context_count: int,
    context_support_threshold: float,
    target_label: int,
    context_label: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    primary, primary_image = read_nifti(segmentation_path(primary_dir, case))
    spacing = spacing_zyx(primary_image)
    shape = np.asarray(primary.shape, dtype=int)
    primary_prob = read_probability_npz(probability_path(primary_dir, case), tuple(shape))
    auxiliary_prob = read_probability_npz(probability_path(auxiliary_dir, case), tuple(shape))

    maximum_label = max(target_label, context_label)
    if primary_prob.shape[0] <= maximum_label or auxiliary_prob.shape[0] <= maximum_label:
        raise ValueError(f"{case}: probability tensor has too few classes")

    extent = np.minimum(np.maximum(np.ceil(roi_mm / spacing).astype(int), 1), shape)
    radius = np.maximum(extent // 2, 1)
    primary_entropy = entropy(primary_prob)
    auxiliary_entropy = entropy(auxiliary_prob)
    primary_target = primary_prob[target_label]
    auxiliary_target = auxiliary_prob[target_label]
    auxiliary_context = auxiliary_prob[context_label]
    all_voxels = np.ones(tuple(shape), dtype=bool)
    context_voxels = (auxiliary_context + auxiliary_target) > context_support_threshold

    maps = [
        ("aux_target_probability", auxiliary_target),
        ("aux_target_uncertainty", auxiliary_target * (1.0 - auxiliary_target)),
        ("aux_entropy", auxiliary_entropy),
        (
            "aux_context_entropy",
            auxiliary_entropy * np.clip(auxiliary_context + auxiliary_target, 0.0, 1.0),
        ),
        ("primary_target_uncertainty", primary_target * (1.0 - primary_target)),
        ("primary_entropy", primary_entropy),
    ]
    proposals: list[tuple[str, int, np.ndarray, float]] = []
    for name, score_map in maps:
        for rank, (center, value) in enumerate(
            top_local_maxima(score_map, maxima_per_map, radius, all_voxels), start=1
        ):
            proposals.append((name, rank, center, value))
    generator = _case_rng(seed, case)
    for rank, (center, value) in enumerate(
        uniform_centers(all_voxels, uniform_count, generator), start=1
    ):
        proposals.append(("uniform", rank, center, value))
    for rank, (center, value) in enumerate(
        uniform_centers(context_voxels, context_count, generator), start=1
    ):
        proposals.append(("context_uniform", rank, center, value))

    reference = None
    if reference_dir is not None:
        reference, _ = read_nifti(segmentation_path(reference_dir, case))
        if reference.shape != primary.shape:
            raise ValueError(f"{case}: reference and prediction shapes differ")
    reference_target = None if reference is None else reference == target_label
    primary_target_mask = primary == target_label
    baseline_dice = (
        np.nan if reference_target is None else dice_score(primary_target_mask, reference_target)
    )
    false_negative = None if reference_target is None else reference_target & ~primary_target_mask

    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[int, ...]] = set()
    for strategy, rank, center, score_value in proposals:
        box = box_from_center(center, extent, shape)
        if box in seen:
            continue
        seen.add(box)
        z0, z1, y0, y1, x0, x1 = box
        volume = (z1 - z0) * (y1 - y0) * (x1 - x0)
        obs_target_stats = region_stats(auxiliary_target, box)
        obs_context_stats = region_stats(auxiliary_context, box)
        obs_entropy_stats = region_stats(auxiliary_entropy, box)
        full_target_stats = region_stats(primary_target, box)
        full_entropy_stats = region_stats(primary_entropy, box)
        row: dict[str, float | int | str] = {
            "case": case,
            "outer_fold": fold,
            "strategy": strategy,
            "rank": rank,
            "score_value": score_value,
            "z0": z0,
            "z1": z1,
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
            "center_z": int(center[0]),
            "center_y": int(center[1]),
            "center_x": int(center[2]),
            "center_z_norm": float(center[0] / max(shape[0] - 1, 1)),
            "center_y_norm": float(center[1] / max(shape[1] - 1, 1)),
            "center_x_norm": float(center[2] / max(shape[2] - 1, 1)),
            "spacing_z": float(spacing[0]),
            "spacing_y": float(spacing[1]),
            "spacing_x": float(spacing[2]),
            "shape_z": int(shape[0]),
            "shape_y": int(shape[1]),
            "shape_x": int(shape[2]),
            "roi_vox_z": int(extent[0]),
            "roi_vox_y": int(extent[1]),
            "roi_vox_x": int(extent[2]),
            "roi_volume_ratio": float(volume / primary.size),
        }
        for prefix, values in (
            ("obs_tumor", obs_target_stats),
            ("obs_pancreas", obs_context_stats),
            ("obs_entropy", obs_entropy_stats),
            ("full_tumor", full_target_stats),
            ("full_entropy", full_entropy_stats),
        ):
            row.update({f"{prefix}_{key}": value for key, value in values.items()})

        if reference_target is not None and false_negative is not None:
            region = np.zeros_like(primary_target_mask, dtype=bool)
            region[z0:z1, y0:y1, x0:x1] = True
            corrected = primary_target_mask | (reference_target & region)
            roi_target = int((reference_target & region).sum())
            roi_fn = int((false_negative & region).sum())
            gain = dice_score(corrected, reference_target) - float(baseline_dice)
            row.update(
                {
                    "oracle_dice_gain": gain,
                    "miss_recovery": int(
                        reference_target.any() and not primary_target_mask.any() and roi_target > 0
                    ),
                    "fn_coverage": float(roi_fn / max(int(false_negative.sum()), 1)),
                    "tumor_coverage": float(roi_target / max(int(reference_target.sum()), 1)),
                }
            )
        rows.append(row)
    return rows


def assign_training_labels(
    table: pd.DataFrame,
    high_gain_percentile: float = 0.8,
    absolute_gain_threshold: float = 0.1,
) -> pd.DataFrame:
    if "oracle_dice_gain" not in table:
        return table
    table = table.copy()
    percentile = table.groupby("case")["oracle_dice_gain"].rank(pct=True, method="average")
    table["case_gain_rank_pct"] = percentile
    table["high_gain_label"] = (
        (table["miss_recovery"] == 1)
        | (percentile >= high_gain_percentile)
        | (table["oracle_dice_gain"] >= absolute_gain_threshold)
    ).astype(np.int8)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--auxiliary-dir", required=True)
    parser.add_argument("--reference-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--roi-mm", nargs=3, type=float, default=(40.0, 48.0, 48.0))
    parser.add_argument("--maxima-per-map", type=int, default=8)
    parser.add_argument("--uniform-samples", type=int, default=8)
    parser.add_argument("--context-samples", type=int, default=8)
    parser.add_argument("--context-support-threshold", type=float, default=0.05)
    parser.add_argument("--high-gain-percentile", type=float, default=0.8)
    parser.add_argument("--absolute-gain-threshold", type=float, default=0.1)
    parser.add_argument("--target-label", type=int, default=2)
    parser.add_argument("--context-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    splits = read_splits(args.splits_json)
    fold_map = validation_fold_map(splits)
    rows: list[dict[str, float | int | str]] = []
    for index, case in enumerate(sorted(fold_map), start=1):
        print(f"[{index}/{len(fold_map)}] {case}")
        rows.extend(
            build_case_candidates(
                case=case,
                fold=fold_map[case],
                primary_dir=Path(args.primary_dir),
                auxiliary_dir=Path(args.auxiliary_dir),
                reference_dir=Path(args.reference_dir) if args.reference_dir else None,
                roi_mm=np.asarray(args.roi_mm, dtype=np.float32),
                maxima_per_map=args.maxima_per_map,
                uniform_count=args.uniform_samples,
                context_count=args.context_samples,
                context_support_threshold=args.context_support_threshold,
                target_label=args.target_label,
                context_label=args.context_label,
                seed=args.seed,
            )
        )
    table = assign_training_labels(
        pd.DataFrame(rows),
        high_gain_percentile=args.high_gain_percentile,
        absolute_gain_threshold=args.absolute_gain_threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote {len(table)} candidates for {table['case'].nunique()} cases to {output}")


if __name__ == "__main__":
    main()
