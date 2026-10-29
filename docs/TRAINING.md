# Training the MSD pipeline

Run substantial training on your own configured compute resources. The commands below are instructions; installation and tests do not start observer training.

## 1. Prepare data and nested splits

```bash
medrefocus prepare-msd --source data/Task07_Pancreas --output data/Dataset007_Pancreas
medrefocus split --dataset data/Dataset007_Pancreas --output data/splits_nested.json
```

For existing patient assignments, add `--outer-splits data/original_outer_splits.json`. The 281-patient cohort gives outer test sizes 57,56,56,56,56. The remaining patients form each fold's development set with internal fit/selection roles.

## 2. Train Global and Auxiliary observers

Observer preprocessing and training use the [official nnU-Net workflow](https://github.com/MIC-DKFZ/nnUNet/blob/v2.5.1/documentation/how_to_use_nnunet.md). The examples target **nnunetv2 2.5.1** and its `.pth` checkpoint names; use a separate compatible nnU-Net environment. Install nnU-Net there and this repository without replacing its PyTorch build:

```bash
python -m pip install nnunetv2==2.5.1
python -m pip install -e .
export nnUNet_raw="$PWD/data/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnUNet_preprocessed"
export nnUNet_results="$PWD/runs/observers/global"
```

The shell examples use Bash; on Windows set the same variables in PowerShell. For outer fold 0, create a separate dataset containing **only its outer development labels**:

```bash
medrefocus outer-dataset --dataset data/Dataset007_Pancreas \
  --splits data/splits_nested.json --fold 0 \
  --output "$nnUNet_raw/Dataset701_PancreasOuter0"
nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity
cp "$nnUNet_raw/Dataset701_PancreasOuter0/internal_splits.json" \
  "$nnUNet_preprocessed/Dataset701_PancreasOuter0/splits_final.json"
python scripts/train_observer.py --dataset 701 --configuration 3d_fullres \
  --seed 2026 --results runs/observers/global
python scripts/train_observer.py --dataset 701 --configuration 3d_fullres \
  --seed 2027 --results runs/observers/auxiliary
```

Here nnU-Net's internal **fold 0** is the one fit/selection split inside **outer fold 0**. The helper runs the standard trainer with independent process seeds and result roots. Global and Auxiliary are independently fitted full-resolution observers in this runnable configuration. An already trained Auxiliary can also be supplied through native probability exports.

The copied [`splits_final.json`](https://github.com/MIC-DKFZ/nnUNet/blob/v2.5.1/documentation/manual_data_splits.md) is installed before training. Dataset fingerprints, CT intensity statistics and spatial plans are computed from the outer development dataset. nnU-Net supplies nonzero cropping, CT clipping/normalization, spacing resampling, patch sampling, augmentation, Dice+CE training and internal checkpointing. Keep observer plans and versions with your run.

Export both observers over `imagesTr` and `imagesTs` using the same selected checkpoint. For example, with `nnUNet_results` set to the Global result root:

```bash
nnUNetv2_predict -i "$nnUNet_raw/Dataset701_PancreasOuter0/imagesTr" \
  -o upstream/fold0/global -d 701 -c 3d_fullres -f 0 \
  -chk checkpoint_best.pth --save_probabilities --disable_tta
nnUNetv2_predict -i "$nnUNet_raw/Dataset701_PancreasOuter0/imagesTs" \
  -o upstream/fold0/global -d 701 -c 3d_fullres -f 0 \
  -chk checkpoint_best.pth --save_probabilities --disable_tta
```

Set `nnUNet_results` to `runs/observers/auxiliary` and repeat with output `upstream/fold0/auxiliary`. The standard [exporter](https://github.com/MIC-DKFZ/nnUNet/blob/v2.5.1/nnunetv2/inference/export_prediction.py) restores cropping, resampling and transposition before saving probabilities. Use the original SimpleITK coordinate convention throughout. For outer folds 1–4, use distinct dataset IDs 702–705, corresponding patient splits, and `upstream/fold1`–`fold4`; independently train each fold's observers.

## 3. Fit UCRP, Local Expert and PCSC

```bash
medrefocus manifest --dataset data/Dataset007_Pancreas \
  --splits data/splits_nested.json --fold 0 --upstream upstream/fold0 \
  --output data/fold0.csv
medrefocus check --manifest data/fold0.csv --training
medrefocus train --manifest data/fold0.csv --config configs/msd_pancreas.yaml \
  --output runs/fold0 --fold 0 --device cuda
```

The default configuration fixes the 40×48×48 mm ROI, one action, 37/118 feature schemas, 120 epochs for UCRP and expert, 0.3/0.6 proposal thresholds, and a 1,000-tree PCSC forest with cutoff 0.5. The ranker uses BCE + 0.5 MSE + within-patient pairwise loss, with 256 pair-sampling attempts per update. Expert training uses weighted CE and foreground Dice; batch padding is excluded from both losses. Selected patches containing no reference tumor are capped at 0.3 times the positive-patch count.

Only fit patients update neural weights; selection patients select UCRP/expert checkpoints. PCSC fits proposals from the outer development partition. The outer test rows are removed before training reads any data. Thresholds are explicit fixed defaults; any new operating-point search belongs inside the outer training partition and must be completed before testing.

Outputs include `ucrp.pt`, `expert.pt`, `pcsc.joblib`, `config.json`, histories and `training_contract.json` with model hashes. Training refuses a nonempty output directory. Nonfinite loss/gradients preserve a failure state and stop the run.

## 4. Freeze, predict and evaluate

```bash
medrefocus predict --manifest data/fold0.csv --models runs/fold0 \
  --role test --output runs/fold0/final --device cuda
medrefocus evaluate --manifest data/fold0.csv --predictions runs/fold0/final \
  --output runs/fold0/evaluation
```

Run each outer fold separately. For an aggregate held-out evaluation, concatenate the five **test-only** manifest subsets (one row per patient), collect their final masks in one directory, and evaluate that 281-row manifest. The default bootstrap resamples patients conditional on fitted models. It does not repeat the complete training procedure.
