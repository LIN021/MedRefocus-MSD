"""NIfTI and nnU-Net probability I/O with explicit array conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_nifti(path: str | Path) -> tuple[np.ndarray, sitk.Image]:
    """Read a NIfTI file as a Z-Y-X NumPy array and return its metadata image."""
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), image


def spacing_zyx(image: sitk.Image) -> np.ndarray:
    return np.asarray(image.GetSpacing()[::-1], dtype=np.float32)


def write_nifti_like(array: np.ndarray, reference: sitk.Image, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(array)
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(target))


def read_probability_npz(
    path: str | Path, spatial_shape: tuple[int, int, int] | None = None
) -> np.ndarray:
    """Read a class-first C-Z-Y-X probability tensor from an nnU-Net NPZ."""
    with np.load(path) as data:
        for key in ("probabilities", "softmax", "predicted_probabilities", "probs", "arr_0"):
            if key in data:
                array = np.asarray(data[key], dtype=np.float32)
                break
        else:
            keys = list(data.keys())
            if len(keys) != 1:
                raise ValueError(f"Cannot identify probability array in {path}; keys={keys}")
            array = np.asarray(data[keys[0]], dtype=np.float32)

    if array.ndim != 4:
        raise ValueError(f"Expected a 4D probability tensor in {path}, got {array.shape}")
    if spatial_shape is not None:
        if tuple(array.shape[1:]) == tuple(spatial_shape):
            pass
        elif tuple(array.shape[:3]) == tuple(spatial_shape):
            array = np.moveaxis(array, -1, 0)
        else:
            raise ValueError(
                f"Probability shape {array.shape} does not match spatial shape {spatial_shape}"
            )
    elif array.shape[0] > 16 and array.shape[-1] <= 16:
        array = np.moveaxis(array, -1, 0)
    return array


def probability_path(directory: str | Path, case: str) -> Path:
    path = Path(directory) / f"{case}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def segmentation_path(directory: str | Path, case: str) -> Path:
    directory = Path(directory)
    for suffix in (".nii.gz", ".nii"):
        path = directory / f"{case}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No segmentation found for {case} in {directory}")
