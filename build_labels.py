"""
Build the cross-sensor label table.

Question the model learns to answer:
    Given what VIIRS saw in a MODIS-sized footprint, did MODIS detect fire there?

Anchor = one VIIRS detection.
Label   = 1 if any MODIS detection falls within MATCH_RADIUS_KM and
          +/- MATCH_WINDOW_MIN of it, else 0.

Features describe the VIIRS pixel plus the neighbouring VIIRS pixels that fall
inside the same MODIS-sized footprint (+/- a few minutes, i.e. same overpass).
That aggregate is our proxy for "what MODIS's pixel would have integrated".

Usage:
    python3 build_labels.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from firms_client import load_raw

EARTH_R = 6371.0

START = date(2023, 2, 10)
END = date(2023, 2, 15)      # exclusive

MATCH_RADIUS_KM = 1.5        # MODIS 1 km pixel + geolocation slack
MATCH_WINDOW_MIN = 30        # same-ish overpass

NEIGH_RADIUS_KM = 1.5        # footprint used to aggregate VIIRS neighbours
NEIGH_WINDOW_MIN = 5         # VIIRS pixels from the same overpass

# MODIS's swath (2330 km) is narrower than VIIRS's (3060 km), so VIIRS detections
# near its own swath edge are often outside MODIS coverage entirely. Without a
# filter the model learns "big scan angle -> no fire", which is a coverage
# artifact, not fire behaviour. Set e.g. 0.5 to reduce it; None keeps everything.
MAX_SCAN_KM: float | None = None

OUT = Path(__file__).parent / "data" / "labels"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# geometry / time helpers
# --------------------------------------------------------------------------- #

def to_xyz(lat, lon) -> np.ndarray:
    """Lat/lon -> unit-sphere Cartesian. Lets a KD-tree measure great circles."""
    la = np.radians(np.asarray(lat, dtype=float))
    lo = np.radians(np.asarray(lon, dtype=float))
    return np.column_stack([np.cos(la) * np.cos(lo),
                            np.cos(la) * np.sin(lo),
                            np.sin(la)])


def ball_radius(km: float) -> float:
    """Chord length on a unit sphere for a given great-circle radius."""
    return 2.0 * np.sin(km / (2.0 * EARTH_R))


def minutes_since_epoch(df: pd.DataFrame) -> np.ndarray:
    """acq_date + acq_time (HHMM) -> minutes, so time is one comparable number."""
    df = df[df["acq_time"].notna()]
    t = df["acq_time"].astype(int).astype(str).str.zfill(4)
    days = (pd.to_datetime(df["acq_date"]) - pd.Timestamp("1970-01-01")).dt.days
    hh = t.str[:2].astype(int)
    mm = t.str[2:].astype(int)
    return (days * 1440 + hh * 60 + mm).to_numpy()


def neighbour_hits(anchor_xyz, anchor_min, pts_xyz, pts_min,
                   radius_km, window_min):
    """Yield, per anchor, the indices of points inside the radius AND time window."""
    tree = cKDTree(pts_xyz)
    hits = tree.query_ball_point(anchor_xyz, ball_radius(radius_km))
    for i, idx in enumerate(hits):
        if not idx:
            yield np.empty(0, dtype=int)
            continue
        idx = np.asarray(idx, dtype=int)
        yield idx[np.abs(pts_min[idx] - anchor_min[i]) <= window_min]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"Loading {START} .. {END} (exclusive)")
    modis, viirs = load_raw(START, END)
    if not len(modis) or not len(viirs):
        raise SystemExit("no data returned")

    modis = modis[modis["acq_time"].notna()].reset_index(drop=True)
    viirs = viirs[viirs["acq_time"].notna()].reset_index(drop=True)

    if MAX_SCAN_KM is not None:
        before = len(viirs)
        viirs = viirs[viirs["scan"] <= MAX_SCAN_KM].reset_index(drop=True)
        print(f"  scan filter <= {MAX_SCAN_KM} km: {before:,} -> {len(viirs):,} VIIRS rows")

    m_xyz, m_min = to_xyz(modis.latitude, modis.longitude), minutes_since_epoch(modis)
    v_xyz, v_min = to_xyz(viirs.latitude, viirs.longitude), minutes_since_epoch(viirs)

    # ---- label: did MODIS detect here? -------------------------------- #
    label = np.zeros(len(viirs), dtype=int)
    for i, hit in enumerate(
        neighbour_hits(v_xyz, v_min, m_xyz, m_min, MATCH_RADIUS_KM, MATCH_WINDOW_MIN)
    ):
        label[i] = int(hit.size > 0)

    # ---- features: aggregate the VIIRS pixels sharing the footprint --- #
    n_neigh = np.zeros(len(viirs), dtype=int)
    frp_sum = np.zeros(len(viirs))
    frp_max = np.zeros(len(viirs))
    ti4_mean = np.zeros(len(viirs))
    dtime = np.zeros(len(viirs), dtype=int)   # spread of the overpass

    for i, hit in enumerate(
        neighbour_hits(v_xyz, v_min, v_xyz, v_min, NEIGH_RADIUS_KM, NEIGH_WINDOW_MIN)
    ):
        n_neigh[i] = hit.size
        if hit.size:
            frp_sum[i] = viirs["frp"].to_numpy()[hit].sum()
            frp_max[i] = viirs["frp"].to_numpy()[hit].max()
            ti4_mean[i] = viirs["bright_ti4"].to_numpy()[hit].mean()
            dtime[i] = v_min[hit].max() - v_min[hit].min()

    conf_map = {"l": 0, "n": 1, "h": 2}

    labels = pd.DataFrame({
        "obs_id": np.arange(len(viirs)),
        "acq_date": viirs["acq_date"].to_numpy(),
        "acq_time": viirs["acq_time"].astype(int).to_numpy(),
        "latitude": viirs["latitude"].to_numpy(),
        "longitude": viirs["longitude"].to_numpy(),
        "satellite": viirs["satellite"].to_numpy(),
        "daynight": viirs["daynight"].to_numpy(),
        # own-pixel features
        "frp": viirs["frp"].to_numpy(),
        "bright_ti4": viirs["bright_ti4"].to_numpy(),
        "bright_ti5": viirs["bright_ti5"].to_numpy(),
        "scan": viirs["scan"].to_numpy(),
        "track": viirs["track"].to_numpy(),
        "conf_ord": viirs["confidence"].map(conf_map).astype("float"),
        # footprint features (the nesting, as numbers)
        "n_neigh": n_neigh,
        "frp_sum": frp_sum,
        "frp_max": frp_max,
        "ti4_mean": ti4_mean,
        "overpass_spread_min": dtime,
        # label
        "modis_detected": label,
    })
    labels["is_night"] = (labels["daynight"] == "N").astype(int)
    labels["doy"] = pd.to_datetime(labels["acq_date"]).dt.day_of_year
    labels = labels.drop(columns=["daynight"])

    path = OUT / f"labels_{START}_{END}.csv"
    labels.to_csv(path, index=False)

    # ---- diagnostics --------------------------------------------------- #
    pos = labels["modis_detected"].mean()
    print(f"\nrows: {len(labels):,}   positives: {labels.modis_detected.sum():,} "
          f"({pos:.1%})   negatives: {(1 - pos):.1%}")

    print("\nmean feature value by label:")
    cols = ["frp", "frp_sum", "frp_max", "bright_ti4", "n_neigh"]
    print(labels.groupby("modis_detected")[cols].mean().round(2).to_string())

    print("\npositives by overpass neighbourhood size:")
    print(labels.groupby("n_neigh", observed=True)["modis_detected"]
          .agg(["mean", "size"]).head(12).round(3).to_string())

    print(f"\nsaved -> {path.relative_to(Path(__file__).parent)}")
    print(labels.head(3).to_string())


if __name__ == "__main__":
    main()
