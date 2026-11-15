"""Label-table and feature construction (pure functions, no I/O)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS_DIR, Settings
from .config import DEFAULT as DEFAULT_SETTINGS
from .geo import SpatialTimeIndex, block_id
from .timeutils import to_minutes

CONF_MAP = {"l": 0, "n": 1, "h": 2}

FEATURE_COLUMNS = [
    # own pixel
    "frp", "bright_ti4", "bright_ti5", "scan", "track", "conf_ord",
    # footprint
    "n_neigh", "frp_sum", "frp_max", "ti4_mean", "overpass_spread_min",
    # context
    "is_night", "doy",
]

META_COLUMNS = [
    "obs_id", "acq_date", "acq_time", "latitude", "longitude",
    "satellite", "source_product", "block",
]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["acq_time"].notna()].reset_index(drop=True)


def _column(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _numeric(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(_column(df, name), errors="coerce").to_numpy(dtype=float)


def build_viirs_features(
    viirs: pd.DataFrame,
    settings: Settings = DEFAULT_SETTINGS,
    id_column: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build one row of model features per VIIRS detection.

    This is the single feature path shared by training and serving. The
    footprint aggregates use *all* VIIRS detections in the supplied frame, not
    only the cluster members, so callers must pass the complete VIIRS set for
    the run's date range and bounding box.

    Returns the feature frame and the per-row acquisition time in minutes.
    """
    viirs = _clean(viirs)
    if settings.max_scan_km is not None and "scan" in viirs.columns:
        viirs = viirs[viirs["scan"] <= settings.max_scan_km].reset_index(drop=True)
    if not len(viirs):
        raise ValueError("need at least one VIIRS detection")

    v_min = to_minutes(viirs["acq_date"], viirs["acq_time"])

    # Aggregate VIIRS pixels sharing the MODIS-sized footprint. This is the
    # nesting expressed as numbers: MODIS responds to the total heat in its
    # pixel, not to a single VIIRS pixel.
    idx = SpatialTimeIndex(viirs["latitude"], viirs["longitude"], v_min)
    frp = _numeric(viirs, "frp")
    ti4 = _numeric(viirs, "bright_ti4")

    n_neigh = np.zeros(len(viirs), dtype=int)
    frp_sum = np.zeros(len(viirs))
    frp_max = np.zeros(len(viirs))
    ti4_mean = np.zeros(len(viirs))
    spread = np.zeros(len(viirs), dtype=int)

    for i, hit in enumerate(
        idx.query(viirs["latitude"], viirs["longitude"], v_min,
                  settings.neigh_radius_km, settings.neigh_window_min)
    ):
        n_neigh[i] = hit.size
        if hit.size:
            frp_sum[i] = frp[hit].sum()
            frp_max[i] = frp[hit].max()
            ti4_mean[i] = ti4[hit].mean()
            spread[i] = v_min[hit].max() - v_min[hit].min()

    source_product = _column(viirs, "source_product", "VIIRS")
    satellite = _column(viirs, "satellite", "unknown")
    daynight = _column(viirs, "daynight", "")

    out = pd.DataFrame({
        "obs_id": np.arange(len(viirs)),
        "acq_date": viirs["acq_date"].to_numpy(),
        "acq_time": viirs["acq_time"].astype("int64").to_numpy(),
        "latitude": viirs["latitude"].to_numpy(),
        "longitude": viirs["longitude"].to_numpy(),
        "satellite": np.asarray(satellite),
        "source_product": np.asarray(source_product),
        "frp": frp,
        "bright_ti4": ti4,
        "bright_ti5": _numeric(viirs, "bright_ti5"),
        "scan": _numeric(viirs, "scan"),
        "track": _numeric(viirs, "track"),
        "conf_ord": _column(viirs, "confidence").map(CONF_MAP).astype("float").to_numpy(),
        "n_neigh": n_neigh,
        "frp_sum": frp_sum,
        "frp_max": frp_max,
        "ti4_mean": ti4_mean,
        "overpass_spread_min": spread,
        "is_night": (np.asarray(daynight) == "N").astype(int),
        "doy": pd.to_datetime(viirs["acq_date"]).dt.day_of_year.to_numpy(),
    })

    if id_column is not None:
        if id_column not in viirs.columns:
            raise ValueError(f"id_column {id_column!r} is not present in the frame")
        out["observation_id"] = viirs[id_column].to_numpy()

    return out, v_min


def build_label_table(
    modis: pd.DataFrame,
    viirs: pd.DataFrame,
    settings: Settings = DEFAULT_SETTINGS,
) -> pd.DataFrame:
    """One row per VIIRS detection; label = did MODIS detect in the same footprint?

    Anchoring on VIIRS (which detects ~5x more than MODIS) yields both classes
    without fabricating negatives.
    """
    modis = _clean(modis)
    viirs = _clean(viirs)
    if not len(modis) or not len(viirs):
        raise ValueError("need both MODIS and VIIRS detections to build labels")

    features, v_min = build_viirs_features(viirs, settings)

    m_min = to_minutes(modis["acq_date"], modis["acq_time"])
    label = SpatialTimeIndex(modis["latitude"], modis["longitude"], m_min).any_within(
        features["latitude"], features["longitude"], v_min,
        settings.match_radius_km, settings.match_window_min,
    )

    features["modis_detected"] = label
    features["block"] = block_id(
        features["latitude"], features["longitude"], settings.block_deg
    )
    return features


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

def labels_path(start: date, end: date, directory: Path = LABELS_DIR) -> Path:
    return directory / f"labels_{start.isoformat()}_{(end).isoformat()}.parquet"


def save_labels(df: pd.DataFrame, start: date, end: date,
                directory: Path = LABELS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = labels_path(start, end, directory)
    df.to_parquet(path, index=False)
    return path
