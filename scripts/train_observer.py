"""Seed one standard nnU-Net observer run without modifying nnU-Net source."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--results", required=True)
    args = p.parse_args()
    root = Path(args.results).resolve()
    # Global and Auxiliary have separate roots, each containing multiple outer datasets.
    if list(root.glob(f"Dataset{int(args.dataset):03d}_*/*/fold_0")):
        raise FileExistsError("This observer dataset/fold already has a run")
    root.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_results"] = str(root)
    from medrefocus.common import seed_everything

    seed_everything(args.seed)
    from nnunetv2.run.run_training import run_training_entry

    sys.argv = ["nnUNetv2_train", args.dataset, args.configuration, "0", "--npz"]
    run_training_entry()
    (root / f"dataset_{args.dataset}_seed.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
