"""Patient manifests and common physical-space validation."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from medrefocus.io import read_nifti, read_probability_npz, spacing_zyx

REQUIRED = ["case", "image", "global_prediction", "global_probabilities", "auxiliary_probabilities"]


def read_manifest(path, training=False):
    path = Path(path).resolve()
    table = pd.read_csv(path, dtype={"case": str})
    required = REQUIRED + (["reference", "role"] if training else [])
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if table.empty or table["case"].isna().any() or table["case"].duplicated().any():
        raise ValueError("Manifest must contain unique nonempty patient identifiers")
    if not table["case"].str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("Patient identifiers may only contain letters, digits, _ and -")
    if training:
        if not set(table["role"]) <= {"fit", "selection", "test"}:
            raise ValueError("Unknown patient role")
        if not {"fit", "selection"} <= set(table["role"]):
            raise ValueError("Fitting and inner-selection patients are required")
    for name in required:
        if name in ["case", "role"]:
            continue
        if table[name].isna().any():
            raise ValueError(f"Missing path in {name}")
        table[name] = table[name].map(
            lambda s: (
                str((path.parent / str(s)).resolve())
                if not Path(str(s)).is_absolute()
                else str(Path(str(s)))
            )
        )
    return table


def assert_geometry(first, second):
    if first.GetSize() != second.GetSize():
        raise ValueError("NIfTI sizes differ")
    for a, b in [
        (first.GetSpacing(), second.GetSpacing()),
        (first.GetOrigin(), second.GetOrigin()),
        (first.GetDirection(), second.GetDirection()),
    ]:
        if not np.allclose(a, b, rtol=0, atol=1e-4):
            raise ValueError("NIfTI physical geometry differs")


def load_case(row, reference=False):
    ct, geometry = read_nifti(row["image"])
    global_mask, global_geometry = read_nifti(row["global_prediction"])
    assert_geometry(geometry, global_geometry)
    if ct.ndim != 3 or not np.isfinite(ct).all():
        raise ValueError("CT must be a finite 3D array")
    if not set(np.unique(global_mask)) <= {0, 1, 2}:
        raise ValueError("Expected Global labels 0, 1, 2")
    pg = read_probability_npz(row["global_probabilities"], ct.shape)
    pa = read_probability_npz(row["auxiliary_probabilities"], ct.shape)
    for p in [pg, pa]:
        if p.shape[0] != 3 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Expected finite three-class probabilities in [0,1]")
        if not np.allclose(p.sum(0), 1.0, atol=2e-3):
            raise ValueError("Probabilities must sum to one on the native grid")
    out = dict(
        ct=ct,
        global_mask=global_mask,
        pg=pg,
        pa=pa,
        geometry=geometry,
        spacing=spacing_zyx(geometry),
    )
    if reference:
        ref, ref_geometry = read_nifti(row["reference"])
        assert_geometry(geometry, ref_geometry)
        if not set(np.unique(ref)) <= {0, 1, 2}:
            raise ValueError("Expected reference labels 0, 1, 2")
        out["reference"] = ref
    return out
