"""Patch dataset and variable-size collation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LocalPatchDataset(Dataset):
    def __init__(self, index: pd.DataFrame, augment: bool = False) -> None:
        self.index = index.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        row = self.index.iloc[item]
        with np.load(Path(row["file"])) as data:
            inputs = np.asarray(data["x"], dtype=np.float32)
            target = np.asarray(data["y"], dtype=np.int64)
        if self.augment:
            for axis in (1, 2, 3):
                if np.random.random() < 0.5:
                    inputs = np.flip(inputs, axis=axis).copy()
                    target = np.flip(target, axis=axis - 1).copy()
        return {
            "x": torch.from_numpy(inputs),
            "y": torch.from_numpy(target),
            "case": str(row["case"]),
        }


def pad_collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor]:
    spatial = (
        np.max(np.asarray([item["y"].shape for item in batch], dtype=int), axis=0)
        .clip(min=8)
        .tolist()
    )
    inputs = torch.zeros((len(batch), 4, *spatial), dtype=torch.float32)
    target = torch.full((len(batch), *spatial), 255, dtype=torch.long)
    valid = torch.zeros((len(batch), *spatial), dtype=torch.bool)
    for index, item in enumerate(batch):
        x = item["x"]
        y = item["y"]
        z, height, width = y.shape
        inputs[index, :, :z, :height, :width] = x
        target[index, :z, :height, :width] = y
        valid[index, :z, :height, :width] = True
    return {"x": inputs, "y": target, "valid": valid}
