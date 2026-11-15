"""Command line entrypoint: python -m terrawatch <command>."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from . import backend_api, config
from .artifacts import default_version, latest_artifact, load_artifact, save_artifact
from .config import Settings
from .evaluate import (ablation, feature_importance, grouped_cv, leak_warnings,
                       logistic_coefficients, optimism, random_split_eval)
from .features import DEFAULT_SET, FEATURE_SETS, TARGET, add_features
from .ingest import FirmsClient
from .labels import build_label_table, labels_path, save_labels
from .models import build_model
from .predict import AGGREGATIONS, predict_bundle, read_bundle, write_predictions


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _date(value: str) -> date:
    return date.fromisoformat(value)


def _settings(args) -> Settings:
    changes = {}
    if getattr(args, "max_scan", None) is not None:
        changes["max_scan_km"] = args.max_scan
    if getattr(args, "seed", None) is not None:
        changes["seed"] = args.seed
    if getattr(args, "block_deg", None) is not None:
        changes["block_deg"] = args.block_deg
    if getattr(args, "bbox", None):
        changes["bbox"] = tuple(float(v) for v in args.bbox.split(","))
    return dataclasses.replace(config.DEFAULT, **changes)


def _summarise(frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        if len(df):
            print(f"  {name:8s} {len(df):8,d} rows  "
                  f"{df['acq_date'].min()} -> {df['acq_date'].max()}")
        else:
            print(f"  {name:8s} (empty)")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_fetch(args) -> None:
    settings = _settings(args)
    config.ensure_dirs()
    print(f"Fetching {args.start} .. {args.end} (exclusive)")
    frames = FirmsClient(settings).fetch(args.start, args.end)
    _summarise(frames)


def cmd_labels(args) -> None:
    settings = _settings(args)
    config.ensure_dirs()
    frames = FirmsClient(settings).load(args.start, args.end)
    if any(not len(df) for df in frames.values()):
        raise SystemExit("no cached data — run `fetch` first")

    print(f"Building labels {args.start} .. {args.end}")
    _summarise(frames)
    labels = build_label_table(frames["modis"], frames["viirs"], settings)
    path = save_labels(labels, args.start, args.end)

    pos = labels[TARGET].mean()
    print(f"\nrows: {len(labels):,}   positives: {int(labels[TARGET].sum()):,} "
          f"({pos:.1%})   blocks: {labels['block'].nunique():,}")
    cols = ["frp", "frp_sum", "frp_max", "bright_ti4", "n_neigh"]
    print("\nmean feature value by label:")
    print(labels.groupby(TARGET)[cols].mean().round(2).to_string())
    print(f"\nsaved -> {path.relative_to(config.ROOT)}")


def _load_labels(args) -> pd.DataFrame:
    path = labels_path(args.start, args.end)
    if not path.exists():
        raise SystemExit(f"no labels at {path} — run `labels` first")
    return add_features(pd.read_parquet(path))


def cmd_train(args) -> None:
    settings = _settings(args)
    df = _load_labels(args)
    set_name = args.feature_set or DEFAULT_SET
    features = FEATURE_SETS[set_name]

    print(f"rows: {len(df):,}   positives: {int(df[TARGET].sum())} "
          f"({df[TARGET].mean():.1%})   blocks: {df['block'].nunique():,}")
    print(f"feature set: {set_name} ({len(features)} features)")

    rnd = random_split_eval(df, features, seed=settings.seed)
    honest = grouped_cv(df, features, seed=settings.seed)

    print("\n=== RANDOM SPLIT (optimistic) ===")
    print(rnd.round(3).to_string())
    print("\n=== GROUPED 5-FOLD CV (honest) ===")
    print(honest.round(3).to_string())
    print("\n=== optimism introduced by the random split ===")
    print(optimism(rnd, honest).to_string())

    print("\n=== feature ablation (grouped CV) ===")
    print(ablation(df, FEATURE_SETS, seed=settings.seed)
          .round(3).to_string(index=False))

    gbm = build_model("lightgbm", features, settings.seed).fit(df[features], df[TARGET])
    print("\n=== lightgbm feature importance (splits) ===")
    print(feature_importance(gbm, features).to_string())
    flagged = leak_warnings(gbm, features)
    if flagged:
        print(f"\n  WARNING: top driver(s) are not physics features: {flagged}")

    lr = build_model("logistic", features, settings.seed).fit(df[features], df[TARGET])
    print("\n=== logistic coefficients (standardised) ===")
    print(logistic_coefficients(lr, features).round(2).to_string())

    version = getattr(args, "model_version", None) or default_version()
    training = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "rows": int(len(df)),
        "positives": int(df[TARGET].sum()),
        "blocks": int(df["block"].nunique()),
        "target": TARGET,
        "bbox": list(settings.bbox),
    }
    metrics = {
        "random_split": rnd.round(6).to_dict(orient="index"),
        "grouped_cv": honest.round(6).to_dict(orient="index"),
        "optimism": optimism(rnd, honest).round(6).to_dict(orient="index"),
    }
    artifact = save_artifact(
        gbm, features, set_name, settings,
        version=version, training=training, metrics=metrics,
    )
    print(f"\nsaved model -> {artifact.relative_to(config.ROOT)}")
    print(f"model version: {version}")


def _model_path(args) -> Path:
    requested = getattr(args, "model", None)
    if requested:
        return Path(requested)
    latest = latest_artifact()
    if latest is None:
        raise SystemExit(
            "no model artifact found; run `train` first or pass --model PATH"
        )
    return latest


def _print_prediction_summary(result: dict) -> None:
    metrics = result.get("metrics", {})
    print(json.dumps(metrics, indent=2))
    print(f"model: {result.get('model')}")
    print(f"predictions: {len(result.get('predictions', []))}")


def cmd_predict(args) -> None:
    model_path = _model_path(args)
    artifact = load_artifact(model_path)
    bundle = read_bundle(args.bundle)
    result = predict_bundle(bundle, artifact, aggregation=args.aggregation)
    out = write_predictions(result, args.out)
    print(f"loaded model -> {model_path}")
    print(f"loaded bundle -> {args.bundle}")
    _print_prediction_summary(result)
    print(f"wrote -> {out}")


def cmd_predict_backend(args) -> None:
    model_path = _model_path(args)
    artifact = load_artifact(model_path)
    bundle = backend_api.build_bundle(
        args.api,
        run_id=args.run_id,
        start=args.start,
        end=args.end,
        bbox=args.bbox,
    )
    print(
        f"run {bundle['run_id']}: {len(bundle['observations'])} VIIRS observations, "
        f"{len(bundle['clusters'])} clusters"
    )
    if args.bundle_out:
        bundle_out = Path(args.bundle_out)
        bundle_out.parent.mkdir(parents=True, exist_ok=True)
        bundle_out.write_text(
            json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote bundle -> {bundle_out}")

    result = predict_bundle(bundle, artifact, aggregation=args.aggregation)
    out = write_predictions(result, args.out)
    print(f"loaded model -> {model_path}")
    _print_prediction_summary(result)
    print(f"wrote -> {out}")

    if args.import_predictions:
        token = args.admin_token or os.environ.get("ADMIN_TOKEN")
        response = backend_api.import_predictions(
            args.api, int(bundle["run_id"]), result, token=token
        )
        print("import response:")
        print(json.dumps(response, indent=2))


def cmd_all(args) -> None:
    cmd_fetch(args)
    print()
    cmd_labels(args)
    print()
    cmd_train(args)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", type=_date, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=_date, required=True, help="YYYY-MM-DD (exclusive)")
    p.add_argument("--max-scan", type=float, default=None,
                   help="drop VIIRS detections wider than this (km)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--block-deg", type=float, default=None)
    p.add_argument("--bbox", type=str, default=None, help="west,south,east,north")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terrawatch")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("fetch", cmd_fetch, "download and cache FIRMS data"),
        ("labels", cmd_labels, "build the label table from cache"),
        ("train", cmd_train, "train and evaluate models"),
        ("all", cmd_all, "fetch -> labels -> train"),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
        if name in ("train", "all"):
            p.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default=None)
            p.add_argument("--model-version", default=None,
                           help="artifact version label (default: timestamped)")
        p.set_defaults(func=handler)

    predict = sub.add_parser("predict", help="score a saved backend bundle")
    predict.add_argument("--bundle", required=True, help="JSON bundle from the backend")
    predict.add_argument("--model", default=None,
                         help="artifact path (default: newest in data/models)")
    predict.add_argument("--out", default="predictions.json")
    predict.add_argument("--aggregation", choices=AGGREGATIONS, default="max")
    predict.set_defaults(func=cmd_predict)

    backend = sub.add_parser(
        "predict-backend",
        help="read a run from the Go API, predict, and optionally import",
    )
    backend.add_argument("--api", default="http://localhost:8000")
    backend.add_argument("--run-id", type=int, default=None,
                         help="default: latest completed run")
    backend.add_argument("--start", default=None, help="override run start filter")
    backend.add_argument("--end", default=None, help="override run end filter")
    backend.add_argument("--bbox", default=None, help="west,south,east,north")
    backend.add_argument("--model", default=None,
                         help="artifact path (default: newest in data/models)")
    backend.add_argument("--out", default="predictions.json")
    backend.add_argument("--bundle-out", default=None,
                         help="optional path to save the fetched prediction bundle")
    backend.add_argument("--aggregation", choices=AGGREGATIONS, default="max")
    backend.add_argument("--import", dest="import_predictions", action="store_true",
                         help="POST predictions to the backend after scoring")
    backend.add_argument("--admin-token", default=None,
                         help="default: ADMIN_TOKEN environment variable")
    backend.set_defaults(func=cmd_predict_backend)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
