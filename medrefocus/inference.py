"""One expert call per case; no reference annotations enter inference."""

from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from medrefocus.data import load_case, read_manifest
from medrefocus.io import write_nifti_like
from medrefocus.local_expert.model import LocalUNet3D
from medrefocus.local_expert.patches import normalize_image
from medrefocus.pcsc.features import extract_features
from medrefocus.pcsc.feature_names import FEATURE_NAMES as PCSC_FEATURES
from medrefocus.ucrp.candidates import build_case_candidates
from medrefocus.ucrp.features import matrix_from_table, FEATURE_NAMES
from medrefocus.ucrp.model import UCRPNetwork


def candidates(row, config, reference=False):
    case = str(row["case"])
    for key, suffix in [
        ("global_prediction", ".nii.gz"),
        ("global_probabilities", ".npz"),
        ("auxiliary_probabilities", ".npz"),
    ]:
        if Path(row[key]).name != case + suffix:
            raise ValueError(f"{key} must be named {case}{suffix}")
    if Path(row["global_prediction"]).parent != Path(row["global_probabilities"]).parent:
        raise ValueError("Global mask and probabilities must share one directory")
    if reference and Path(row["reference"]).name != case + ".nii.gz":
        raise ValueError("Reference must be named <case>.nii.gz")
    return pd.DataFrame(
        build_case_candidates(
            case,
            0,
            Path(row["global_prediction"]).parent,
            Path(row["auxiliary_probabilities"]).parent,
            Path(row["reference"]).parent if reference else None,
            np.asarray(config["roi_mm"]),
            config["maxima_per_map"],
            config["uniform_samples"],
            config["context_samples"],
            config["context_support_threshold"],
            2,
            1,
            config["seed"],
        )
    )


def select_region(table, checkpoint, model, device):
    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise ValueError("UCRP feature contract mismatch")
    x = (matrix_from_table(table) - checkpoint["mean"]) / checkpoint["std"]
    with torch.inference_mode():
        scores = model(torch.from_numpy(x).to(device)).sigmoid().cpu().numpy()
    row = table.iloc[int(np.argmax(scores))].copy()
    row["originating_map_value"] = row["score_value"]
    row["ucrp_score"] = float(scores.max())
    # PCSC's three planning-score aliases contain the selected MLP score.
    row["score_value"] = row["ucrp_score"]
    return row


def make_proposal(case, row, expert, config, device):
    box = tuple(
        slice(int(row[a]), int(row[b])) for a, b in [("z0", "z1"), ("y0", "y1"), ("x0", "x1")]
    )
    global_mask = case["global_mask"]
    roi = global_mask[box]
    x = np.stack(
        [
            normalize_image(case["ct"])[box],
            case["pg"][1][box],
            case["pg"][2][box],
            roi.astype(np.float32) / 2,
        ]
    ).astype(np.float32)
    tensor = torch.from_numpy(x[None]).to(device)
    pads = tuple(v for n in reversed(x.shape[1:]) for v in (0, max(0, 8 - n)))
    tensor = F.pad(tensor, pads)
    with torch.inference_mode():
        probability = (
            expert(tensor).softmax(1)[0, 2, : x.shape[1], : x.shape[2], : x.shape[3]].cpu().numpy()
        )
    if not np.isfinite(probability).all():
        raise FloatingPointError("Nonfinite expert output")
    current = bool((roi == 2).any())
    threshold = (
        config["expert"]["existing_threshold"] if current else config["expert"]["missing_threshold"]
    )
    mask = probability >= threshold
    proposal = global_mask.copy()
    proposal[box][mask] = 2
    row = row.copy()
    row["local_threshold"] = threshold
    row["local_has_current_tumor"] = int(current)
    row["local_applied"] = int(mask.any())
    if np.any((global_mask == 2) & (proposal != 2)):
        raise AssertionError("An existing tumor voxel was removed")
    return proposal, probability, row


def positive_probability(model, table):
    """Handle all-negative/all-positive small fitting sets without column errors."""
    probabilities = model.predict_proba(table)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(table))


class MedRefocus:
    def __init__(self, model_dir, device="cpu"):
        self.device = torch.device(device)
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf8"))
        self.ranker_state = torch.load(
            model_dir / "ucrp.pt", map_location="cpu", weights_only=False
        )
        self.ranker = UCRPNetwork(
            len(FEATURE_NAMES), self.config["ucrp"]["hidden_units"], self.config["ucrp"]["dropout"]
        ).to(self.device)
        self.ranker.load_state_dict(self.ranker_state["model"])
        self.ranker.eval()
        state = torch.load(model_dir / "expert.pt", map_location="cpu", weights_only=False)
        self.expert = LocalUNet3D(base=self.config["expert"]["base_width"]).to(self.device)
        self.expert.load_state_dict(state["model"])
        self.expert.eval()
        self.gate = joblib.load(model_dir / "pcsc.joblib")
        if self.gate["feature_names"] != PCSC_FEATURES:
            raise ValueError("PCSC feature contract mismatch")

    def predict(self, record):
        # Explicitly drop a reference path even if a training manifest was supplied.
        row = {k: v for k, v in dict(record).items() if k != "reference"}
        case = load_case(row, reference=False)
        selected = select_region(
            candidates(row, self.config), self.ranker_state, self.ranker, self.device
        )
        proposal, probability, selected = make_proposal(
            case, selected, self.expert, self.config, self.device
        )
        features = extract_features(
            selected, case["global_mask"], proposal, probability, case["spacing"], 2
        )
        x = pd.DataFrame([features], columns=PCSC_FEATURES).replace([np.inf, -np.inf], np.nan)
        score = float(positive_probability(self.gate["model"], x)[0])
        accept = score >= self.config["pcsc"]["threshold"]
        final = proposal if accept else case["global_mask"].copy()
        info = dict(
            case=row["case"],
            expert_calls=1,
            pcsc_score=score,
            accepted=bool(accept),
            roi_zyx=[int(selected[k]) for k in ["z0", "z1", "y0", "y1", "x0", "x1"]],
            threshold=float(selected["local_threshold"]),
            added_tumor_voxels=int(((final == 2) & (case["global_mask"] != 2)).sum()),
            reference_used=False,
        )
        return final, case["geometry"], info


def predict_manifest(manifest, model_dir, output_dir, device="cpu", role=None):
    table = read_manifest(manifest)
    if role is not None:
        if "role" not in table:
            raise ValueError("Requested role is not in manifest")
        table = table[table["role"] == role]
    if table.empty:
        raise ValueError("No patients selected for prediction")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = MedRefocus(model_dir, device)
    records = []
    for row in table.to_dict("records"):
        final, geometry, record = model.predict(row)
        write_nifti_like(final, geometry, output_dir / (row["case"] + ".nii.gz"))
        records.append(record)
    (output_dir / "inference_records.json").write_text(
        json.dumps(records, indent=2), encoding="utf8"
    )
    return records
