"""Prepare MSD files, patient splits and per-outer-fold input manifests."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from medrefocus.common import read_splits


def prepare_msd(source, destination):
    source, destination = Path(source), Path(destination)
    cases = sorted(p.name[:-7] for p in (source / "labelsTr").glob("*.nii.gz"))
    if not cases:
        raise ValueError("Expected Task07_Pancreas/imagesTr and labelsTr")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    (destination / "imagesTr").mkdir()
    (destination / "labelsTr").mkdir()
    for case in cases:
        image = sitk.ReadImage(str(source / "imagesTr" / (case + ".nii.gz")))
        if image.GetDimension() == 4:
            if image.GetSize()[3] != 1:
                raise ValueError(f"{case}: expected a single CT channel")
            image = image[:, :, :, 0]
        if image.GetDimension() != 3:
            raise ValueError(f"{case}: expected 3D CT")
        sitk.WriteImage(image, str(destination / "imagesTr" / (case + "_0000.nii.gz")))
        shutil.copy2(source / "labelsTr" / (case + ".nii.gz"), destination / "labelsTr")
    metadata = dict(
        channel_names={"0": "CT"},
        labels={"background": 0, "pancreas": 1, "tumor": 2},
        numTraining=len(cases),
        file_ending=".nii.gz",
    )
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return cases


def make_splits(dataset, output, seed=2026, existing=None):
    cases = sorted(p.name[:-7] for p in (Path(dataset) / "labelsTr").glob("*.nii.gz"))
    if len(cases) < 10:
        raise ValueError("At least ten patients are required for five outer folds")
    if existing:
        outer = read_splits(existing)
        if len(outer) != 5:
            raise ValueError("Expected five existing outer folds")
        test = [c for split in outer for c in split["val"]]
        if len(test) != len(set(test)) or set(test) != set(cases):
            raise ValueError("Every patient must appear in exactly one outer test fold")
        for split in outer:
            if set(split["train"]) != set(cases) - set(split["val"]):
                raise ValueError("Outer training partition must complement its test fold")
    else:
        chunks = np.array_split(np.random.default_rng(seed).permutation(cases), 5)
        outer = [
            dict(train=sorted(set(cases) - set(chunk)), val=sorted(chunk.tolist()))
            for chunk in chunks
        ]
    nested = []
    for fold, split in enumerate(outer):
        development = np.random.default_rng(seed + fold).permutation(split["train"])
        n_selection = max(1, math.ceil(0.1 * len(development)))
        nested.append(
            dict(
                fold=fold,
                fit=sorted(development[n_selection:].tolist()),
                selection=sorted(development[:n_selection].tolist()),
                test=sorted(split["val"]),
            )
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(seed=seed, folds=nested), indent=2), encoding="utf8")
    return nested


def outer_dataset(dataset, splits, fold, destination):
    """nnU-Net receives only this outer fold's development patients."""
    dataset, destination = Path(dataset), Path(destination)
    split = json.loads(Path(splits).read_text(encoding="utf8"))["folds"][fold]
    if destination.exists():
        raise FileExistsError(destination)
    for subdir in ["imagesTr", "labelsTr", "imagesTs"]:
        (destination / subdir).mkdir(parents=True, exist_ok=True)
    for role in ["fit", "selection", "test"]:
        for case in split[role]:
            shutil.copy2(
                dataset / "imagesTr" / (case + "_0000.nii.gz"),
                destination / ("imagesTs" if role == "test" else "imagesTr"),
            )
            if role != "test":
                shutil.copy2(dataset / "labelsTr" / (case + ".nii.gz"), destination / "labelsTr")
    metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf8"))
    metadata["numTraining"] = len(split["fit"]) + len(split["selection"])
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    internal = [dict(train=split["fit"], val=split["selection"])]
    (destination / "internal_splits.json").write_text(
        json.dumps(internal, indent=2), encoding="utf8"
    )
    return internal


def make_manifest(dataset, splits, fold, upstream, output):
    dataset, upstream = Path(dataset).resolve(), Path(upstream).resolve()
    split = json.loads(Path(splits).read_text(encoding="utf8"))["folds"][fold]
    rows = []
    for role in ["fit", "selection", "test"]:
        for case in split[role]:
            rows.append(
                dict(
                    case=case,
                    role=role,
                    image=str(dataset / "imagesTr" / (case + "_0000.nii.gz")),
                    reference=str(dataset / "labelsTr" / (case + ".nii.gz")),
                    global_prediction=str(upstream / "global" / (case + ".nii.gz")),
                    global_probabilities=str(upstream / "global" / (case + ".npz")),
                    auxiliary_probabilities=str(upstream / "auxiliary" / (case + ".npz")),
                )
            )
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return rows
