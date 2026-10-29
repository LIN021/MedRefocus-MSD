"""Fit UCRP, the local expert and PCSC inside one outer training partition."""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from medrefocus.common import seed_everything
from medrefocus.data import load_case, read_manifest
from medrefocus.inference import candidates, make_proposal, select_region
from medrefocus.local_expert.data import LocalPatchDataset, pad_collate
from medrefocus.local_expert.model import LocalUNet3D, soft_dice_loss
from medrefocus.local_expert.patches import normalize_image
from medrefocus.metrics import acceptance_label
from medrefocus.pcsc.feature_names import FEATURE_NAMES as PCSC_FEATURES
from medrefocus.pcsc.features import extract_features
from medrefocus.ucrp.candidates import assign_training_labels
from medrefocus.ucrp.features import FEATURE_NAMES, matrix_from_table
from medrefocus.ucrp.model import UCRPNetwork


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def checked_step(loss, model, optimizer, clip, output, batch):
    """Fail on nonfinite training and preserve state; never skip or retry a batch."""
    finite = bool(torch.isfinite(loss))
    if finite:
        loss.backward()
        finite = all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
        )
    if not finite:
        torch.save(
            dict(
                model=_cpu(model.state_dict()),
                optimizer=_cpu(optimizer.state_dict()),
                batch=_cpu(batch),
                torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
            ),
            output / "nonfinite_failure.pt",
        )
        raise FloatingPointError("Nonfinite training state; saved nonfinite_failure.pt and stopped")
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()


def pair_indices(table, attempts, rng, margin):
    groups = [
        g.index.to_numpy()
        for _, g in table.groupby("case", sort=True)
        if len(g) > 1 and g.oracle_dice_gain.max() - g.oracle_dice_gain.min() > margin
    ]
    hi, lo = [], []
    if not groups:
        return hi, lo
    for _ in range(attempts):
        a, b = rng.choice(groups[int(rng.integers(len(groups)))], size=2, replace=False)
        delta = table.loc[a, "oracle_dice_gain"] - table.loc[b, "oracle_dice_gain"]
        if abs(delta) <= margin:
            continue
        hi.append(int(a if delta > 0 else b))
        lo.append(int(b if delta > 0 else a))
    return hi, lo


def fit_ranker(table, config, output, device):
    seed_everything(config["seed"])
    cfg = config["ucrp"]
    rng = np.random.default_rng(config["seed"])
    fit = table[table.role == "fit"].reset_index(drop=True)
    val = table[table.role == "selection"].reset_index(drop=True)
    x = matrix_from_table(fit)
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-6] = 1
    x = torch.from_numpy((x - mean) / std).to(device)
    vx = torch.from_numpy((matrix_from_table(val) - mean) / std).to(device)
    y = torch.tensor(fit.high_gain_label.to_numpy(np.float32), device=device)
    gain = torch.tensor(np.clip(fit.oracle_dice_gain.to_numpy(np.float32), 0, 1), device=device)
    model = UCRPNetwork(len(FEATURE_NAMES), cfg["hidden_units"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    best = -float("inf")
    state = None
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for inds in np.array_split(
            rng.permutation(len(fit)), max(1, int(np.ceil(len(fit) / cfg["batch_size"])))
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[inds])
            bce = F.binary_cross_entropy_with_logits(logits, y[inds])
            mse = F.mse_loss(logits.sigmoid(), gain[inds])
            hi, lo = pair_indices(fit, cfg["pair_attempts"], rng, cfg["rank_margin"])
            ranking = (
                F.softplus(-(model(x[hi]) - model(x[lo]))).mean() if hi else logits.new_tensor(0.0)
            )
            loss = bce + 0.5 * mse + ranking
            checked_step(
                loss,
                model,
                optimizer,
                cfg["gradient_clip"],
                output,
                dict(x=x[inds], labels=y[inds]),
            )
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            scores = model(vx).sigmoid().cpu().numpy()
        vv = val.copy()
        vv["score"] = scores
        top = (
            vv.sort_values("score", ascending=False, kind="mergesort")
            .groupby("case", sort=False)
            .head(1)
        )
        metric = float(top.oracle_dice_gain.mean() + 0.01 * top.miss_recovery.sum())
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), selection_metric=metric))
        if metric > best:
            best = metric
            state = dict(
                model=_cpu(model.state_dict()),
                mean=mean,
                std=std,
                feature_names=FEATURE_NAMES,
                epoch=epoch,
                selection_metric=metric,
            )
    if state is None:
        raise RuntimeError("No UCRP checkpoint selected")
    torch.save(state, output / "ucrp.pt")
    model.load_state_dict(state["model"])
    model.eval()
    (output / "ucrp_history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
    return model, state


def build_patches(manifest, candidate_tables, ranker, state, config, output, device):
    patch_dir = output / "patches"
    patch_dir.mkdir()
    records = []
    for record in manifest.to_dict("records"):
        case = load_case(record, reference=True)
        selected = select_region(candidate_tables[record["case"]], state, ranker, device)
        box = tuple(
            slice(int(selected[a]), int(selected[b]))
            for a, b in [("z0", "z1"), ("y0", "y1"), ("x0", "x1")]
        )
        x = np.stack(
            [
                normalize_image(case["ct"])[box],
                case["pg"][1][box],
                case["pg"][2][box],
                case["global_mask"][box].astype(np.float32) / 2,
            ]
        ).astype(np.float32)
        y = case["reference"][box].astype(np.uint8)
        file = patch_dir / (record["case"] + ".npz")
        np.savez_compressed(file, x=x, y=y)
        records.append(
            dict(
                case=record["case"],
                role=record["role"],
                file=str(file),
                tumor_voxels=int((y == 2).sum()),
            )
        )
    return pd.DataFrame(records)


def fit_expert(index, config, output, device):
    seed_everything(config["seed"])
    cfg = config["expert"]
    fitting = index[index.role == "fit"]
    positive = fitting[fitting.tumor_voxels > 0]
    negative = fitting[fitting.tumor_voxels == 0]
    if positive.empty:
        raise ValueError("No target-containing fitting patches")
    cap = min(len(negative), round(len(positive) * cfg["negative_to_positive_ratio"]))
    fitting = pd.concat(
        [positive, negative.sample(n=cap, random_state=config["seed"])], ignore_index=True
    )
    validation = index[index.role == "selection"]
    loader = DataLoader(
        LocalPatchDataset(fitting, augment=True),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate,
    )
    valid_loader = DataLoader(
        LocalPatchDataset(validation), batch_size=1, num_workers=0, collate_fn=pad_collate
    )
    model = LocalUNet3D(base=cfg["base_width"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    weights = torch.tensor(cfg["class_weights"], device=device)
    best = -1.0
    state = None
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            valid = batch["valid"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, ignore_index=255) + soft_dice_loss(
                logits, y, valid
            )
            checked_step(loss, model, optimizer, cfg["gradient_clip"], output, batch)
            losses.append(float(loss.detach().cpu()))
        model.eval()
        tp = pred_n = ref_n = 0
        with torch.inference_mode():
            for batch in valid_loader:
                y = batch["y"].to(device)
                valid = batch["valid"].to(device)
                pred = (model(batch["x"].to(device)).argmax(1) == 2) & valid
                ref = (y == 2) & valid
                tp += int((pred & ref).sum())
                pred_n += int(pred.sum())
                ref_n += int(ref.sum())
        dice = 2 * tp / (pred_n + ref_n) if pred_n + ref_n else 1.0
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), selection_tumor_dice=dice))
        if dice > best:
            best = dice
            state = dict(
                model=_cpu(model.state_dict()),
                epoch=epoch,
                selection_tumor_dice=dice,
                base_width=cfg["base_width"],
            )
    if state is None:
        raise RuntimeError("No local-expert checkpoint selected")
    torch.save(state, output / "expert.pt")
    model.load_state_dict(state["model"])
    model.eval()
    (output / "expert_history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
    return model


def train(manifest_path, config, output, device="cpu", fold=0):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory for training")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    manifest = read_manifest(manifest_path, training=True)
    # Remove the entire outer test partition before any training data are loaded.
    development = manifest[manifest.role.isin(["fit", "selection"])].copy()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    candidate_tables = {}
    tables = []
    for record in development.to_dict("records"):
        load_case(record, reference=True)
        table = assign_training_labels(candidates(record, config, reference=True))
        table["role"] = record["role"]
        candidate_tables[record["case"]] = table
        tables.append(table)
    ranker, state = fit_ranker(pd.concat(tables, ignore_index=True), config, output, device)
    index = build_patches(development, candidate_tables, ranker, state, config, output, device)
    expert = fit_expert(index, config, output, device)
    features = []
    labels = []
    weights = []
    for record in development.to_dict("records"):
        case = load_case(record, reference=True)
        selected = select_region(candidate_tables[record["case"]], state, ranker, device)
        proposal, probability, selected = make_proposal(case, selected, expert, config, device)
        features.append(
            extract_features(
                selected, case["global_mask"], proposal, probability, case["spacing"], 2
            )
        )
        label, weight = acceptance_label(
            case["global_mask"], proposal, case["reference"], case["spacing"], config["pcsc"]
        )
        labels.append(label)
        weights.append(weight)
    cfg = config["pcsc"]
    forest = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=cfg["seed_base"] + fold,
        n_jobs=cfg["n_jobs"],
    )
    gate = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("forest", forest),
        ]
    )
    gate.fit(
        pd.DataFrame(features, columns=PCSC_FEATURES).replace([np.inf, -np.inf], np.nan),
        labels,
        forest__sample_weight=weights,
    )
    joblib.dump(dict(model=gate, feature_names=PCSC_FEATURES), output / "pcsc.joblib")
    contracts = dict(
        fit_patients=development.loc[development.role == "fit", "case"].tolist(),
        selection_patients=development.loc[development.role == "selection", "case"].tolist(),
        outer_test_patients=manifest.loc[manifest.role == "test", "case"].tolist(),
        outer_test_reference_loaded=False,
        ucrp_features=len(FEATURE_NAMES),
        pcsc_features=len(PCSC_FEATURES),
        model_sha256={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ["ucrp.pt", "expert.pt", "pcsc.joblib"]
        },
    )
    (output / "training_contract.json").write_text(json.dumps(contracts, indent=2), encoding="utf8")
    return contracts
