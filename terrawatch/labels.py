"""Label-table construction (pure functions, no I/O)."""

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
    if settings.max_scan_km is not None:
        viirs = viirs[viirs["scan"] <= settings.max_scan_km].reset_index(drop=True)
    if not len(modis) or not len(viirs):
        raise ValueError("need both MODIS and VIIRS detections to build labels")

    m_min = to_minutes(modis["acq_date"], modis["acq_time"])
    v_min = to_minutes(viirs["acq_date"], viirs["acq_time"])

    label = SpatialTimeIndex(modis["latitude"], modis["longitude"], m_min).any_within(
        viirs["latitude"], viirs["longitude"], v_min,
        settings.match_radius_km, settings.match_window_min,
    )

    # Aggregate VIIRS pixels sharing the MODIS-sized footprint. This is the
    # nesting expressed as numbers: MODIS responds to the total heat in its
    # pixel, not to a single VIIRS pixel.
    idx = SpatialTimeIndex(viirs["latitude"], viirs["longitude"], v_min)
    frp = viirs["frp"].to_numpy()
    ti4 = viirs["bright_ti4"].to_numpy()

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

    out = pd.DataFrame({
        "obs_id": np.arange(len(viirs)),
        "acq_date": viirs["acq_date"].to_numpy(),
        "acq_time": viirs["acq_time"].astype("int64").to_numpy(),
        "latitude": viirs["latitude"].to_numpy(),
        "longitude": viirs["longitude"].to_numpy(),
        "satellite": viirs["satellite"].to_numpy(),
        "source_product": viirs.get("source_product", pd.Series(["VIIRS"] * len(viirs))),
        "frp": frp,
        "bright_ti4": ti4,
        "bright_ti5": viirs["bright_ti5"].to_numpy(),
        "scan": viirs["scan"].to_numpy(),
        "track": viirs["track"].to_numpy(),
        "conf_ord": viirs["confidence"].map(CONF_MAP).astype("float"),
        "n_neigh": n_neigh,
        "frp_sum": frp_sum,
        "frp_max": frp_max,
        "ti4_mean": ti4_mean,
        "overpass_spread_min": spread,
        "modis_detected": label,
    })
    out["is_night"] = (viirs["daynight"] == "N").astype(int).to_numpy()
    out["doy"] = pd.to_datetime(out["acq_date"]).dt.day_of_year
    out["block"] = block_id(out["latitude"], out["longitude"], settings.block_deg)
    return out


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
