"""
Train the cross-sensor detection model.

Task: given VIIRS observations in a MODIS-sized footprint, predict
      P(MODIS detected fire here).

Three things are reported deliberately:

  1. random split   -> optimistic. Neighbouring pixels of the same fire land on
                       both sides, so the model memorises fires instead of
                       learning sensor behaviour.
  2. grouped CV     -> honest. Whole 1-degree cells are held out, 5 folds.
  3. ablation       -> which features actually carry generalisable signal.

Usage:
    python3 train.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data" / "labels" / "labels_2023-02-10_2023-02-15.csv"
BLOCK_DEG = 1.0
SEED = 42
TARGET = "modis_detected"

# FRP is heavy-tailed, so model its log. Monotonic constraints encode physics:
# +1 means "as this increases, P(detect) may not decrease".
MONOTONE = {
    "log_frp": 1,
    "log_frp_sum": 1,
    "log_frp_max": 1,
    "n_neigh": 1,
    "bright_ti4": 1,
    "bright_ti5": 1,
    "ti4_mean": 1,
    "conf_ord": 1,
    "scan": 0,            # known coverage confounder -> leave free
    "track": 0,
    "is_night": 0,
    "doy": 0,             # meaningless on a 5-day sample, also leaks
    "overpass_spread_min": 0,
}

PHYSICS = ["log_frp", "log_frp_sum", "log_frp_max", "n_neigh",
           "bright_ti4", "bright_ti5", "ti4_mean", "conf_ord"]

FEATURE_SETS = {
    "physics only": PHYSICS,
    "physics + night": PHYSICS + ["is_night"],
    "physics + geom + night": PHYSICS + ["scan", "track", "is_night"],
    "physics + geom + time": PHYSICS + ["scan", "track", "is_night", "doy"],
}

# The set we actually trust: no `doy`, because on a 5-day sample it is a date ID
# and our folds hold out space but not time, so it leaks.
DEFAULT_SET = "physics + geom + night"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("frp", "frp_sum", "frp_max"):
        df[f"log_{c}"] = np.log1p(df[c])
    df["_block"] = (
        np.floor(df.latitude / BLOCK_DEG).astype(int) * 1000
        + np.floor(df.longitude / BLOCK_DEG).astype(int)
    )
    return df


def metrics(y_true, p) -> dict:
    return {
        "PR-AUC": average_precision_score(y_true, p),
        "ROC-AUC": roc_auc_score(y_true, p),
        "Brier": brier_score_loss(y_true, p),
    }


def build_models(feats):
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=7,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            verbosity=-1,
            monotone_constraints=[MONOTONE[f] for f in feats],
            monotone_constraints_method="advanced",
        ),
    }


def grouped_cv(df, feats, n_splits=5) -> pd.DataFrame:
    y, X, g = df[TARGET], df[feats], df["_block"]
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, g))
    out = {name: [] for name in build_models(feats)}
    for tr, te in folds:
        for name, model in build_models(feats).items():
            model.fit(X.iloc[tr], y.iloc[tr])
            out[name].append(metrics(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
    return pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in out.items()}).T


def main() -> None:
    df = add_features(pd.read_csv(DATA)).reset_index(drop=True)
    y = df[TARGET]
    print(f"rows: {len(df):,}   positives: {int(y.sum())} ({y.mean():.1%})   "
          f"blocks: {df['_block'].nunique()}")

    all_feats = FEATURE_SETS[DEFAULT_SET]

    # ---- 1. random split: the leak ------------------------------------- #
    tr, te = train_test_split(df.index, test_size=0.25, random_state=SEED, stratify=y)
    rnd = {}
    for name, model in build_models(all_feats).items():
        model.fit(df[all_feats].iloc[tr], y.iloc[tr])
        rnd[name] = metrics(y.iloc[te], model.predict_proba(df[all_feats].iloc[te])[:, 1])
    rnd = pd.DataFrame(rnd).T
    print(f"\n=== RANDOM SPLIT (optimistic) — test positives {int(y.iloc[te].sum())} ===")
    print(rnd.round(3).to_string())

    # ---- 2. grouped CV: honest ----------------------------------------- #
    print("\n=== GROUPED 5-FOLD CV (honest) ===")
    honest = grouped_cv(df, all_feats)
    print(honest.round(3).to_string())

    print("\n=== optimism introduced by the random split ===")
    print((rnd[["PR-AUC", "ROC-AUC"]] - honest[["PR-AUC", "ROC-AUC"]]).round(3).to_string())

    # ---- 3. ablation --------------------------------------------------- #
    print("\n=== feature ablation (grouped CV, mean over folds) ===")
    print(f"{'feature set':<28}{'model':<10}{'PR-AUC':>8}{'ROC-AUC':>9}{'Brier':>8}")
    for set_name, feats in FEATURE_SETS.items():
        res = grouped_cv(df, feats)
        for model_name in ("logistic", "lightgbm"):
            r = res.loc[model_name]
            print(f"{set_name:<28}{model_name:<10}{r['PR-AUC']:>8.3f}"
                  f"{r['ROC-AUC']:>9.3f}{r['Brier']:>8.3f}")

    # ---- 4. what the model leans on ------------------------------------ #
    model = build_models(all_feats)["lightgbm"].fit(df[all_feats], y)
    imp = pd.Series(model.feature_importances_, index=all_feats).sort_values(ascending=False)
    print("\n=== lightgbm feature importance (splits) ===")
    print(imp.to_string())

    lr = build_models(PHYSICS)["logistic"].fit(df[PHYSICS], y)
    coef = pd.Series(lr[-1].coef_[0], index=PHYSICS).sort_values(ascending=False)
    print("\n=== logistic coefficients, physics only (standardised) ===")
    print(coef.round(2).to_string())


if __name__ == "__main__":
    main()
