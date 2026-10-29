"""Build four-channel local-expert patches from held-out UCRP regions."""

from __future__ import annotations


import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if not len(finite):
        return np.zeros_like(image)
    lower, upper = np.percentile(finite, (0.5, 99.5))
    clipped = np.clip(image, lower, upper)
    scale = float(clipped.std())
    return (clipped - float(clipped.mean())) / max(scale, 1e-6)
