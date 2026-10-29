# Data and coordinate contracts

## Inputs

Acquire the labeled **MSD Task07_Pancreas** training cohort locally. The supplied converter produces `imagesTr/<case>_0000.nii.gz`, `labelsTr/<case>.nii.gz` and `dataset.json`. Labels are background 0, normal pancreas 1 and tumor 2. The converter writes a new directory and leaves the downloaded source untouched.

All downstream arrays are in **Z,Y,X** order. Probabilities are **C,Z,Y,X**, with three channels in the same label order. Use nnU-Net's native-grid probability export, with `probabilities` as the NPZ key. Do not pass preprocessed/cropped or transposed training arrays directly.

For each outer fold, arrange observer exports as:

```text
upstream/fold0/
  global/<case>.nii.gz
  global/<case>.npz
  auxiliary/<case>.npz
```

Global masks, CT and reference masks must share size, spacing, origin and direction. `check` validates these fields and verifies finite, normalized probabilities of the expected shape. NPZ arrays do not carry physical geometry: their exporter must preserve the native coordinate convention. Avoid arbitrary image reorientation between observer export and downstream processing.

## Manifest

A CSV contains one row per patient and these columns:

| Column | Meaning |
|---|---|
| `case` | Unique identifier: letters, digits, `_`, `-` |
| `image` | Native 3D CT |
| `global_prediction` | Native hard Global mask, `<case>.nii.gz` |
| `global_probabilities` | `<case>.npz` in the same directory as the Global mask |
| `auxiliary_probabilities` | Native Auxiliary probabilities, `<case>.npz` |
| `reference` | Reference mask for training/evaluation; absent from inference-only manifests |
| `role` | `fit`, `selection` or `test` for training/evaluation |

Relative paths are resolved against the CSV location. `manifest` writes absolute local paths automatically. A training CSV includes all roles, but `train` removes the outer test partition before loading any image or reference. A prediction CSV can omit references entirely.

## Splits

`split` shuffles sorted patient identifiers deterministically, creates five outer test folds, then reserves 10% (rounded up) of each outer development set for internal selection. With 281 patients this yields 57/56/56/56/56 test cases, 23 selection cases, and 201/202 fitting cases. Every patient appears in exactly one outer test fold.

To retain pre-existing outer folds, pass `--outer-splits path/to/splits_final.json`. The file is a five-entry nnU-Net list with `train` and `val` identifiers. The command validates disjointness and full cohort coverage before creating internal splits. It does not infer historical patient assignments from aggregate fold sizes.

Each fold owns its upstream observers and downstream models. A single set of cross-validated predictions assembled from other outer-fold models is not a substitute for these per-fold inputs.
