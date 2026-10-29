import copy

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import torch
from surface_distance import metrics as surface

from medrefocus.data import assert_geometry, load_case
from medrefocus.inference import candidates, make_proposal, positive_probability
from medrefocus.local_expert.data import pad_collate
from medrefocus.local_expert.model import soft_dice_loss
from medrefocus.metrics import complete_miss, surface_metrics
from medrefocus.pcsc.feature_names import FEATURE_NAMES as PCSC_FEATURES
from medrefocus.pcsc.features import extract_features
from medrefocus.training import checked_step, pair_indices
from medrefocus.ucrp.candidates import entropy
from medrefocus.ucrp.features import FEATURE_NAMES, matrix_from_table


def test_prediction_only_candidates_and_feature_contract(synthetic, tiny_config):
    _, records = synthetic
    record = records[0]
    with_labels = candidates(record, tiny_config, reference=True)
    no_labels = candidates({k: v for k, v in record.items() if k != "reference"}, tiny_config)
    pd.testing.assert_frame_equal(with_labels[no_labels.columns], no_labels)
    assert matrix_from_table(no_labels).shape[1] == len(FEATURE_NAMES) == 37
    assert not any("oracle" in f or "reference" in f for f in FEATURE_NAMES)
    for row in no_labels.to_dict("records"):
        for a, b in [("z0", "z1"), ("y0", "y1"), ("x0", "x1")]:
            assert 0 <= row[a] < row[b] <= 8
    case = load_case(record)
    proposal = case["global_mask"].copy()
    proposal[4, 4, 4] = 2
    selected = no_labels.iloc[0].copy()
    selected["local_threshold"] = 0.3
    selected["local_applied"] = 1
    selected["local_has_current_tumor"] = 0
    selected["ucrp_score"] = 0.7
    selected["score_value"] = 0.7
    features = extract_features(
        selected,
        case["global_mask"],
        proposal,
        np.full((8, 8, 8), 0.4),
        case["spacing"],
        2,
    )
    assert list(features) == PCSC_FEATURES
    assert len(features) == 118
    assert (
        features["planner_ucrp_score"] == features["planner_score_value"] == features["roi_score"]
    )


def test_entropy_zero_and_pair_attempts():
    assert entropy(np.array([1.0, 0.0, 0.0])[:, None, None, None]).item() == 0.0
    table = pd.DataFrame(dict(case=["x"] * 3, oracle_dice_gain=[0.0, 0.0, 0.0]))
    assert pair_indices(table, 256, np.random.default_rng(1), 0.01) == ([], [])
    table.oracle_dice_gain = [0.0, 0.1, 0.3]
    hi, lo = pair_indices(table, 17, np.random.default_rng(1), 0.01)
    assert 0 < len(hi) <= 17
    assert np.all(
        table.loc[hi, "oracle_dice_gain"].to_numpy() > table.loc[lo, "oracle_dice_gain"].to_numpy()
    )


def test_padding_cannot_affect_foreground_dice():
    batch = pad_collate([dict(x=torch.zeros(4, 3, 4, 5), y=torch.zeros(3, 4, 5, dtype=torch.long))])
    assert batch["y"].shape == (1, 8, 8, 8)
    assert torch.all(batch["y"][~batch["valid"]] == 255)
    logits = torch.randn(1, 3, 8, 8, 8)
    changed = logits.clone()
    changed.masked_fill_(~batch["valid"].unsqueeze(1), 100.0)
    a = soft_dice_loss(logits, batch["y"], batch["valid"])
    b = soft_dice_loss(changed, batch["y"], batch["valid"])
    torch.testing.assert_close(a, b)
    # Foreground-only mean: background is not a third Dice term.
    target = torch.ones((1, 1, 1, 1), dtype=torch.long)
    uniform = torch.zeros(1, 3, 1, 1, 1)
    foreground_loss = soft_dice_loss(uniform, target, torch.ones_like(target, dtype=torch.bool))
    assert float(foreground_loss) == pytest.approx(0.75, abs=1e-5)


def test_shifted_geometry_is_rejected():
    image = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    shifted = sitk.Image(image)
    shifted.SetOrigin((0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="geometry"):
        assert_geometry(image, shifted)


def test_one_action_add_only_and_auxiliary_absent_from_expert(synthetic, tiny_config):
    _, rows = synthetic
    case = load_case(rows[1])
    row = candidates(rows[1], tiny_config).iloc[0]

    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            assert x.shape[1] == 4
            assert torch.equal(x[0, 1], torch.from_numpy(case["pg"][1]))
            logits = torch.zeros((1, 3, *x.shape[2:]))
            logits[:, 2, 4, 4, 4] = 20
            return logits

    expert = Expert()
    proposal, _, _ = make_proposal(case, row, expert, tiny_config, "cpu")
    assert expert.calls == 1
    assert np.all(proposal[case["global_mask"] == 2] == 2)
    altered = copy.deepcopy(case)
    altered["pa"] = 1 - case["pa"]
    other, _, _ = make_proposal(altered, row, Expert(), tiny_config, "cpu")
    assert np.array_equal(proposal, other)


def test_area_weighted_surface_metrics_and_fp_only_miss():
    ref = np.zeros((8, 8, 8), bool)
    ref[2:5, 2:5, 2:5] = True
    pred = np.roll(ref, 1, axis=0)
    spacing = (3.0, 1.0, 0.8)
    distances = surface.compute_surface_distances(ref, pred, spacing)
    nsd, hd95 = surface_metrics(pred, ref, spacing)
    assert nsd == surface.compute_surface_dice_at_tolerance(distances, 2.0)
    assert hd95 == surface.compute_robust_hausdorff(distances, 95)
    fp = np.zeros_like(ref)
    fp[7, 7, 7] = True
    assert complete_miss(fp, ref) == 1
    assert surface_metrics(np.zeros_like(ref), ref, spacing) == (0.0, float("inf"))


@pytest.mark.parametrize("classes, expected", [([0], 0.0), ([1], 1.0), ([0, 1], 0.7)])
def test_single_class_forest(classes, expected):
    class Gate:
        classes_ = classes

        def predict_proba(self, table):
            return np.tile([1.0] if len(classes) == 1 else [0.3, 0.7], (len(table), 1))

    assert positive_probability(Gate(), pd.DataFrame({"x": [1]}))[0] == expected


def test_nonfinite_training_stops_and_saves_cpu_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(FloatingPointError):
        checked_step(
            model(torch.ones(1, 2)).sum() * float("nan"),
            model,
            optimizer,
            5,
            tmp_path,
            {"x": torch.ones(1, 2)},
        )
    saved = torch.load(tmp_path / "nonfinite_failure.pt", weights_only=False)
    assert saved["batch"]["x"].device.type == "cpu"
    for key in before:
        torch.testing.assert_close(before[key], model.state_dict()[key])
