"""Shared deterministic and validation helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def read_splits(path: str | Path) -> list[dict[str, list[str]]]:
    with Path(path).open(encoding="utf-8") as handle:
        splits = json.load(handle)
    if not isinstance(splits, list) or not splits:
        raise ValueError("splits_final.json must contain a non-empty list")
    for fold, split in enumerate(splits):
        if set(split) != {"train", "val"}:
            raise ValueError(f"Fold {fold} must contain exactly train and val")
        if set(split["train"]) & set(split["val"]):
            raise ValueError(f"Fold {fold} contains train/validation overlap")
    return splits


def validation_fold_map(splits: list[dict[str, list[str]]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fold, split in enumerate(splits):
        for case in split["val"]:
            if case in result:
                raise ValueError(f"Case {case} appears in multiple validation folds")
            result[str(case)] = fold
    return result


def ensure_columns(table: Any, required: set[str], name: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")
