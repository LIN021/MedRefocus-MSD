from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class UCRPNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(hidden_units) != 2:
            raise ValueError("UCRP requires exactly two hidden layers")
        first, second = (int(value) for value in hidden_units)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.LayerNorm(first),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.LayerNorm(second),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(second, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)
