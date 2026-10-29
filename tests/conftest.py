import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import torch
import yaml


torch.set_num_threads(1)


@pytest.fixture
def tiny_config():
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/msd_pancreas.yaml").read_text())
    config = copy.deepcopy(config)
    config.update(roi_mm=[8.0, 8.0, 8.0], maxima_per_map=1, uniform_samples=1, context_samples=1)
    config["ucrp"].update(epochs=1, hidden_units=[8, 4], batch_size=8)
    config["expert"].update(epochs=1, base_width=2)
    config["pcsc"].update(n_estimators=3, min_samples_leaf=1)
    return config


@pytest.fixture
def synthetic(tmp_path):
    directories = {k: tmp_path / k for k in ["ct", "gt", "global", "auxiliary"]}
    for d in directories.values():
        d.mkdir()
    rows = []
    rng = np.random.default_rng(3)
    for i, role in enumerate(["fit"] * 4 + ["selection"] * 2 + ["test"] * 2):
        case = f"synthetic_{i}"
        ct = rng.normal(50, 30, (8, 8, 8)).astype(np.float32)
        ref = np.zeros((8, 8, 8), np.uint8)
        ref[1:7, 1:7, 1:7] = 1
        ref[3:5, 3:5, 3:5] = 2
        base = np.where(ref == 2, 1, ref).astype(np.uint8)
        if i % 2:
            base[3, 3, 3] = 2
        for key, array in [("ct", ct), ("gt", ref), ("global", base)]:
            image = sitk.GetImageFromArray(array)
            image.SetOrigin((4.0, -8.0, 10.0))
            image.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(image, str(directories[key] / (case + ".nii.gz")))
        pg = np.full((3, 8, 8, 8), 0.05, np.float32)
        for label in range(3):
            pg[label, base == label] = 0.9
        pa = rng.uniform(0.1, 0.9, pg.shape).astype(np.float32)
        pa /= pa.sum(0, keepdims=True)
        np.savez_compressed(directories["global"] / (case + ".npz"), probabilities=pg)
        np.savez_compressed(directories["auxiliary"] / (case + ".npz"), probabilities=pa)
        rows.append(
            dict(
                case=case,
                role=role,
                image=str(directories["ct"] / (case + ".nii.gz")),
                reference=str(directories["gt"] / (case + ".nii.gz")),
                global_prediction=str(directories["global"] / (case + ".nii.gz")),
                global_probabilities=str(directories["global"] / (case + ".npz")),
                auxiliary_probabilities=str(directories["auxiliary"] / (case + ".npz")),
            )
        )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, rows
