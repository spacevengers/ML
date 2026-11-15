"""Command line entrypoint: python -m terrawatch <command>."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import date

import pandas as pd

from . import config
from .config import Settings
from .evaluate import (ablation, feature_importance, grouped_cv, leak_warnings,
                       logistic_coefficients, optimism, random_split_eval)
from .features import DEFAULT_SET, FEATURE_SETS, TARGET, add_features
from .ingest import FirmsClient
from .labels import build_label_table, labels_path, save_labels
from .models import build_model


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
        p.set_defaults(func=handler)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
