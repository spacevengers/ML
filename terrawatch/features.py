"""Feature definitions shared by training and serving."""

from __future__ import annotations

import numpy as np
import pandas as pd

LOG_SOURCES = ("frp", "frp_sum", "frp_max")

# Physics features: all derived from what the sensor actually measured.
PHYSICS = [
    "log_frp", "log_frp_sum", "log_frp_max", "n_neigh",
    "bright_ti4", "bright_ti5", "ti4_mean", "conf_ord",
]

GEOMETRY = ["scan", "track"]
CONTEXT = ["is_night"]

# +1 = "as this rises, P(detect) may not fall". Encodes physics as a prior so the
# model cannot invent nonsense on a small sample.
MONOTONE = {
    "log_frp": 1, "log_frp_sum": 1, "log_frp_max": 1, "n_neigh": 1,
    "bright_ti4": 1, "bright_ti5": 1, "ti4_mean": 1, "conf_ord": 1,
    "scan": 0, "track": 0, "is_night": 0, "doy": 0, "overpass_spread_min": 0,
}

FEATURE_SETS = {
    "physics": PHYSICS,
    "physics+night": PHYSICS + CONTEXT,
    "physics+geom+night": PHYSICS + GEOMETRY + CONTEXT,
    "physics+geom+time": PHYSICS + GEOMETRY + CONTEXT + ["doy"],
}

# The set we trust. `doy` is excluded: on a short sample it acts as a date ID and
# our folds hold out space but not time.
DEFAULT_SET = "physics+geom+night"

TARGET = "modis_detected"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed FRP columns (FRP is heavy-tailed)."""
    df = df.copy()
    for col in LOG_SOURCES:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    return df


def feature_frame(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return df[names]


def monotone_list(names: list[str]) -> list[int]:
    return [MONOTONE[n] for n in names]
