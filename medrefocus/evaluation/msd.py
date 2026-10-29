"""Paired whole-volume tumor evaluation on a specified patient manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medrefocus.data import assert_geometry, read_manifest
from medrefocus.io import read_nifti, spacing_zyx
from medrefocus.metrics import complete_miss, dice_score, surface_metrics


def evaluate(manifest, predictions, output, role="test", resamples=10000, seed=2026):
    table = read_manifest(manifest)
    if "reference" not in table:
        raise ValueError("Evaluation requires reference paths")
    # Resolve evaluation-only paths without exposing them to inference.
    root = Path(manifest).resolve().parent
    if role is not None:
        if "role" not in table:
            raise ValueError("No role column; pass --role all for an inference manifest")
        table = table[table.role == role]
    if table.empty:
        raise ValueError("Empty evaluation cohort")
    rows = []
    for record in table.to_dict("records"):
        ref, geometry = read_nifti(root / record["reference"])
        base, base_geometry = read_nifti(record["global_prediction"])
        final, final_geometry = read_nifti(Path(predictions) / (record["case"] + ".nii.gz"))
        assert_geometry(geometry, base_geometry)
        assert_geometry(geometry, final_geometry)
        for array in [ref, base, final]:
            if not set(np.unique(array)) <= {0, 1, 2}:
                raise ValueError("Expected labels 0/1/2")
        ref, base, final = ref == 2, base == 2, final == 2
        bnsd, bhd = surface_metrics(base, ref, spacing_zyx(geometry))
        fnsd, fhd = surface_metrics(final, ref, spacing_zyx(geometry))
        bdice, fdice = dice_score(base, ref), dice_score(final, ref)
        bmiss, fmiss = complete_miss(base, ref), complete_miss(final, ref)
        rows.append(
            dict(
                case=record["case"],
                base_dice=bdice,
                final_dice=fdice,
                base_nsd=bnsd,
                final_nsd=fnsd,
                base_hd95_mm=bhd,
                final_hd95_mm=fhd,
                base_miss=bmiss,
                final_miss=fmiss,
                recovered=int(bmiss and not fmiss),
                new_miss=int(not bmiss and fmiss),
                recovered_ge_025=int(bmiss and fdice >= 0.25),
                recovered_ge_050=int(bmiss and fdice >= 0.5),
                severe_dice_decline=int(fdice - bdice < -0.05),
                severe_nsd_decline=int(fnsd - bnsd < -0.05),
            )
        )
    patients = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    summary = dict(
        patients=len(patients),
        baseline_misses=int(patients.base_miss.sum()),
        final_misses=int(patients.final_miss.sum()),
        recovered=int(patients.recovered.sum()),
        new_misses=int(patients.new_miss.sum()),
    )
    for metric in ["dice", "nsd"]:
        b = patients["base_" + metric].to_numpy()
        f = patients["final_" + metric].to_numpy()
        valid = np.isfinite(b) & np.isfinite(f)
        delta = (f - b)[valid] * 100
        item = dict(
            baseline_percent=float(np.nanmean(b) * 100),
            final_percent=float(np.nanmean(f) * 100),
            paired_gain_pp=float(delta.mean()),
            paired_patients=int(valid.sum()),
        )
        if resamples and len(delta):
            means = [
                float(rng.choice(delta, len(delta), replace=True).mean()) for _ in range(resamples)
            ]
            item["paired_gain_95ci_pp"] = np.percentile(means, [2.5, 97.5]).tolist()
        summary[metric] = item
    finite = np.isfinite(patients.base_hd95_mm) & np.isfinite(patients.final_hd95_mm)
    summary["hd95_matched_finite_subset"] = dict(
        patients=int(finite.sum()),
        baseline_mm=float(patients.loc[finite, "base_hd95_mm"].mean()) if finite.any() else None,
        final_mm=float(patients.loc[finite, "final_hd95_mm"].mean()) if finite.any() else None,
    )
    for name in [
        "recovered_ge_025",
        "recovered_ge_050",
        "severe_dice_decline",
        "severe_nsd_decline",
    ]:
        summary[name] = int(patients[name].sum())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    patients.to_csv(output / "patients.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf8"
    )
    return summary
