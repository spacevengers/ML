"""Evaluation: metrics, splits, cross-validation, diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split

from .features import MONOTONE, TARGET
from .models import MODEL_NAMES, build_models

METRIC_COLUMNS = ["PR-AUC", "ROC-AUC", "Brier"]


def metrics(y_true, proba) -> dict:
    return {
        "PR-AUC": average_precision_score(y_true, proba),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
    }


def random_split_eval(df: pd.DataFrame, features: list[str],
                      test_size: float = 0.25, seed: int = 42) -> pd.DataFrame:
    """Optimistic baseline. Neighbouring pixels of a fire land on both sides."""
    y = df[TARGET]
    train, test = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=y
    )
    rows = {}
    for name, model in build_models(features, seed).items():
        model.fit(df[features].iloc[train], y.iloc[train])
        proba = model.predict_proba(df[features].iloc[test])[:, 1]
        rows[name] = metrics(y.iloc[test], proba)
    return pd.DataFrame(rows).T[METRIC_COLUMNS]


def grouped_cv(df: pd.DataFrame, features: list[str],
               n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Honest evaluation: whole spatial blocks are held out per fold."""
    y, X, groups = df[TARGET], df[features], df["block"]
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    collected: dict[str, list[dict]] = {name: [] for name in MODEL_NAMES}
    for train, test in folds:
        for name, model in build_models(features, seed).items():
            model.fit(X.iloc[train], y.iloc[train])
            proba = model.predict_proba(X.iloc[test])[:, 1]
            collected[name].append(metrics(y.iloc[test], proba))
    return pd.DataFrame(
        {name: pd.DataFrame(v).mean() for name, v in collected.items()}
    ).T[METRIC_COLUMNS]


def optimism(optimistic: pd.DataFrame, honest: pd.DataFrame) -> pd.DataFrame:
    """How much the random split flatters each model."""
    return (optimistic[["PR-AUC", "ROC-AUC"]]
            - honest[["PR-AUC", "ROC-AUC"]]).round(3)


def ablation(df: pd.DataFrame, feature_sets: dict[str, list[str]],
             models: tuple[str, ...] = ("logistic", "lightgbm"),
             n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Grouped-CV score for each feature set x model combination."""
    rows = []
    for set_name, features in feature_sets.items():
        res = grouped_cv(df, features, n_splits=n_splits, seed=seed)
        for model in models:
            row = res.loc[model]
            rows.append({
                "feature_set": set_name,
                "model": model,
                **{c: row[c] for c in METRIC_COLUMNS},
            })
    return pd.DataFrame(rows)


def feature_importance(model, features: list[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )


def logistic_coefficients(model, features: list[str]) -> pd.Series:
    return pd.Series(model[-1].coef_[0], index=features).sort_values(ascending=False)


def leak_warnings(model, features: list[str], top: int = 3) -> list[str]:
    """Flag when the top features are geometry instead of physics."""
    imp = feature_importance(model, features)
    flagged = [f for f in imp.head(top).index if MONOTONE.get(f) == 0 and f not in ("is_night",)]
    return flagged
