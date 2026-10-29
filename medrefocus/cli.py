"""Command-line entry points for the MSD implementation."""

from __future__ import annotations

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="medrefocus", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-msd", help="Convert downloaded Task07 CT to nnU-Net format")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser(
        "split", help="Create nested patient splits; optionally retain existing outer folds"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--outer-splits")
    p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("outer-dataset", help="Create one outer-development nnU-Net dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("manifest", help="Bind native-grid predictions to one outer fold")
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("check", help="Check files, geometry and probability contracts")
    p.add_argument("--manifest", required=True)
    p.add_argument("--training", action="store_true")
    p = sub.add_parser(
        "train", help="Fit UCRP, expert and PCSC inside one outer-training partition"
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p = sub.add_parser("predict", help="One local action and accept-or-retain per patient")
    p.add_argument("--manifest", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--role", choices=["fit", "selection", "test", "all"], default="all")
    p = sub.add_parser(
        "evaluate", help="Evaluate frozen predictions; references are read only here"
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--role", choices=["fit", "selection", "test", "all"], default="test")
    p.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args(argv)
    from medrefocus import prepare

    if args.command == "prepare-msd":
        result = dict(patients=len(prepare.prepare_msd(args.source, args.output)))
    elif args.command == "split":
        folds = prepare.make_splits(args.dataset, args.output, args.seed, args.outer_splits)
        result = [
            dict(fold=f["fold"], **{k: len(f[k]) for k in ["fit", "selection", "test"]})
            for f in folds
        ]
    elif args.command == "outer-dataset":
        result = prepare.outer_dataset(args.dataset, args.splits, args.fold, args.output)
    elif args.command == "manifest":
        result = dict(
            patients=len(
                prepare.make_manifest(
                    args.dataset, args.splits, args.fold, args.upstream, args.output
                )
            )
        )
    elif args.command == "check":
        from medrefocus.data import load_case, read_manifest

        table = read_manifest(args.manifest, training=args.training)
        for record in table.to_dict("records"):
            load_case(record, reference=args.training and record["role"] != "test")
        result = dict(patients=len(table), geometry="passed")
    elif args.command == "train":
        from medrefocus.common import load_yaml
        from medrefocus.training import train

        result = train(args.manifest, load_yaml(args.config), args.output, args.device, args.fold)
    elif args.command == "predict":
        from medrefocus.inference import predict_manifest

        result = predict_manifest(
            args.manifest,
            args.models,
            args.output,
            args.device,
            None if args.role == "all" else args.role,
        )
    else:
        from medrefocus.evaluation.msd import evaluate

        result = evaluate(
            args.manifest,
            args.predictions,
            args.output,
            None if args.role == "all" else args.role,
            args.bootstrap,
        )
    print(json.dumps(result, indent=2))
