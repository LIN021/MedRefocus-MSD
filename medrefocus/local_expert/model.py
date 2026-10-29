"""Compact 3D U-Net used for focused local re-observation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class LocalUNet3D(nn.Module):
    """Three-level U-Net; base width 24 gives the paper's focused expert."""

    def __init__(self, input_channels: int = 4, classes: int = 3, base: int = 24) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(input_channels, base)
        self.encoder2 = ConvBlock(base, base * 2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.decoder2 = ConvBlock(base * 6, base * 2)
        self.decoder1 = ConvBlock(base * 3, base)
        self.output = nn.Conv3d(base, classes, 1)

    @staticmethod
    def _resize_like(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if inputs.shape[2:] == reference.shape[2:]:
            return inputs
        return F.interpolate(
            inputs, size=reference.shape[2:], mode="trilinear", align_corners=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(inputs)
        second = self.encoder2(self.pool(first))
        bottleneck = self.bottleneck(self.pool(second))
        decoded2 = self._resize_like(bottleneck, second)
        decoded2 = self.decoder2(torch.cat([decoded2, second], dim=1))
        decoded1 = self._resize_like(decoded2, first)
        decoded1 = self.decoder1(torch.cat([decoded1, first], dim=1))
        return self.output(decoded1)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    target_safe = target.clone()
    target_safe[~valid] = 0
    one_hot = F.one_hot(target_safe, num_classes=logits.shape[1]).movedim(-1, 1).float()
    valid_channel = valid.unsqueeze(1)
    probabilities = probabilities * valid_channel
    one_hot = one_hot * valid_channel
    dimensions = (0, 2, 3, 4)
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dimensions)
    denominator = probabilities[:, 1:].sum(dimensions) + one_hot[:, 1:].sum(dimensions)
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()
