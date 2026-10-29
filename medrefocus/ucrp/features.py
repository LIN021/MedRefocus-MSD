"""Uncertainty-Conditioned Recoverability Planning feature contract."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


CONTINUOUS_FEATURES = [
    "score_value",
    "rank",
    "roi_volume_ratio",
    "center_z_norm",
    "center_y_norm",
    "center_x_norm",
    "roi_vox_z",
    "roi_vox_y",
    "roi_vox_x",
    "obs_tumor_mean",
    "obs_tumor_max",
    "obs_tumor_p90",
    "obs_tumor_sum",
    "obs_pancreas_mean",
    "obs_pancreas_max",
    "obs_pancreas_p90",
    "obs_pancreas_sum",
    "obs_entropy_mean",
    "obs_entropy_max",
    "obs_entropy_p90",
    "obs_entropy_sum",
    "full_tumor_mean",
    "full_tumor_max",
    "full_tumor_p90",
    "full_tumor_sum",
    "full_entropy_mean",
    "full_entropy_max",
    "full_entropy_p90",
    "full_entropy_sum",
]

STRATEGIES = [
    "aux_target_probability",
    "aux_target_uncertainty",
    "aux_entropy",
    "aux_context_entropy",
    "primary_target_uncertainty",
    "primary_entropy",
    "uniform",
    "context_uniform",
]

FEATURE_NAMES = CONTINUOUS_FEATURES + [f"strategy_{name}" for name in STRATEGIES]
FORBIDDEN_PATTERN = re.compile(r"gt|reference|fn|coverage|oracle|dice|miss|label|gain", re.I)


def add_strategy_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for strategy in STRATEGIES:
        table[f"strategy_{strategy}"] = (table["strategy"] == strategy).astype(np.float32)
    return table


def validate_feature_contract(feature_names: list[str] = FEATURE_NAMES) -> None:
    leaked = [name for name in feature_names if FORBIDDEN_PATTERN.search(name)]
    if leaked:
        raise ValueError(f"Reference-derived planner inputs are forbidden: {leaked}")


def matrix_from_table(table: pd.DataFrame) -> np.ndarray:
    validate_feature_contract()
    table = add_strategy_columns(table)
    missing = set(FEATURE_NAMES) - set(table.columns)
    if missing:
        raise ValueError(f"Candidate table is missing planner features: {sorted(missing)}")
    return (
        table[FEATURE_NAMES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
