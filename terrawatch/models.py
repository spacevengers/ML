"""Model registry. Adding a model is one entry in ``build_models``."""

from __future__ import annotations

import lightgbm as lgb
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import monotone_list

MODEL_NAMES = ("dummy", "logistic", "lightgbm")


def build_model(name: str, features: list[str], seed: int = 42):
    """Instantiate an unfitted estimator for the given feature list."""
    if name == "dummy":
        return DummyClassifier(strategy="prior")

    if name == "logistic":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        )

    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=7,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=-1,
            monotone_constraints=monotone_list(features),
            monotone_constraints_method="advanced",
        )

    raise ValueError(f"unknown model: {name!r} (choose from {MODEL_NAMES})")


def build_models(features: list[str], seed: int = 42) -> dict[str, object]:
    return {name: build_model(name, features, seed) for name in MODEL_NAMES}
