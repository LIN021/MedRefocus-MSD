import json

import numpy as np
import pandas as pd
import SimpleITK as sitk

from medrefocus.evaluation.msd import evaluate
from medrefocus.inference import MedRefocus, predict_manifest
from medrefocus.prepare import make_splits, outer_dataset, prepare_msd
from medrefocus.training import train


def test_full_cpu_fit_freeze_predict_evaluate(synthetic, tiny_config, tmp_path, monkeypatch):
    manifest, records = synthetic
    training_table = pd.read_csv(manifest)
    # Training must not even try to open these outer-test references.
    training_table.loc[training_table.role == "test", "reference"] = str(
        tmp_path / "unavailable.nii.gz"
    )
    training_manifest = tmp_path / "training.csv"
    training_table.to_csv(training_manifest, index=False)
    model_dir = tmp_path / "models"
    contract = train(training_manifest, tiny_config, model_dir, device="cpu")
    assert not contract["outer_test_reference_loaded"]
    assert len(contract["fit_patients"]) == 4
    assert len(contract["selection_patients"]) == 2
    assert len(contract["outer_test_patients"]) == 2
    assert set(contract["model_sha256"]) == {"ucrp.pt", "expert.pt", "pcsc.joblib"}

    model = MedRefocus(model_dir, "cpu")
    # Force reject: the complete label map must be preserved voxel for voxel.
    model.config["pcsc"]["threshold"] = 2.0
    final, geometry, info = model.predict(training_table.iloc[-1])
    base_image = sitk.ReadImage(records[-1]["global_prediction"])
    assert np.array_equal(final, sitk.GetArrayFromImage(base_image))
    assert geometry.GetOrigin() == base_image.GetOrigin()
    assert not info["accepted"]
    assert info["expert_calls"] == 1
    assert not info["reference_used"]

    # Inference-only manifests do not need a reference column.
    inference_table = training_table[training_table.role == "test"].drop(columns="reference")
    inference_csv = tmp_path / "inference.csv"
    inference_table.to_csv(inference_csv, index=False)
    output = tmp_path / "final"
    predictions = predict_manifest(inference_csv, model_dir, output, "cpu")
    assert len(predictions) == 2
    assert all(p["expert_calls"] == 1 and not p["reference_used"] for p in predictions)
    summary = evaluate(manifest, output, tmp_path / "evaluation", resamples=20)
    assert summary["patients"] == 2
    assert summary["new_misses"] == 0  # add-only + retain cannot lose existing overlap
    assert (tmp_path / "evaluation" / "patients.csv").exists()
    json.loads((tmp_path / "evaluation" / "summary.json").read_text())


def test_nested_split_counts_and_outer_data_exclusion(tmp_path):
    source = tmp_path / "prepared"
    (source / "labelsTr").mkdir(parents=True)
    for i in range(281):
        (source / "labelsTr" / f"synthetic_{i:03}.nii.gz").touch()
    splits = make_splits(source, tmp_path / "splits.json")
    assert [len(s["test"]) for s in splits] == [57, 56, 56, 56, 56]
    assert [len(s["selection"]) for s in splits] == [23] * 5
    tests = [c for s in splits for c in s["test"]]
    assert len(tests) == len(set(tests)) == 281
    for split in splits:
        assert not (set(split["fit"]) & set(split["selection"]))
        assert not ((set(split["fit"]) | set(split["selection"])) & set(split["test"]))


def test_prepare_and_inner_dataset(synthetic, tmp_path):
    _, records = synthetic
    # A miniature MSD-format source, written only under the test directory.
    source = tmp_path / "Task07"
    for sub in ["imagesTr", "labelsTr"]:
        (source / sub).mkdir(parents=True)
    import shutil

    for record in records:
        shutil.copy2(record["image"], source / "imagesTr" / (record["case"] + ".nii.gz"))
        shutil.copy2(record["reference"], source / "labelsTr" / (record["case"] + ".nii.gz"))
    dataset = tmp_path / "dataset"
    assert len(prepare_msd(source, dataset)) == 8
    custom = dict(
        folds=[
            dict(
                fold=0,
                fit=[r["case"] for r in records[:4]],
                selection=[r["case"] for r in records[4:6]],
                test=[r["case"] for r in records[6:]],
            )
        ]
    )
    splits = tmp_path / "custom.json"
    splits.write_text(json.dumps(custom))
    destination = tmp_path / "outer"
    outer_dataset(dataset, splits, 0, destination)
    assert len(list((destination / "labelsTr").glob("*.nii.gz"))) == 6
    assert len(list((destination / "imagesTs").glob("*.nii.gz"))) == 2
    for record in records[6:]:
        assert not (destination / "labelsTr" / (record["case"] + ".nii.gz")).exists()
