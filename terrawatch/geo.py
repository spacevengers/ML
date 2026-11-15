"""Geospatial helpers: distances, spatial indexing, grid blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R_KM = 6371.0


def to_xyz(lat, lon) -> np.ndarray:
    """Lat/lon (degrees) -> unit-sphere Cartesian.

    Lets a KD-tree measure great-circle distance: chord length on the unit
    sphere is monotonic in arc length.
    """
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def chord_for_km(km: float) -> float:
    """Chord length on the unit sphere for a given great-circle radius."""
    return float(2.0 * np.sin(km / (2.0 * EARTH_R_KM)))


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(
        lambda v: np.radians(np.asarray(v, dtype=float)), (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def block_id(lat, lon, deg: float = 1.0) -> np.ndarray:
    """Integer id for a lat/lon grid cell, used for spatial hold-out."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return (
        np.floor(lat / deg).astype(np.int64) * 100_000
        + np.floor(lon / deg).astype(np.int64)
    )


class SpatialTimeIndex:
    """Query points that fall inside a radius *and* a time window.

    Built once over the reference set; queried once per anchor.
    """

    def __init__(self, lat, lon, minutes):
        self.xyz = to_xyz(lat, lon)
        self.minutes = np.asarray(minutes)
        self._tree = cKDTree(self.xyz)

    def query(self, lat, lon, minutes, radius_km: float, window_min: int):
        """Yield, per query point, the indices of matches."""
        anchor_xyz = to_xyz(lat, lon)
        anchor_min = np.asarray(minutes)
        hits = self._tree.query_ball_point(anchor_xyz, chord_for_km(radius_km))
        for i, idx in enumerate(hits):
            if not idx:
                yield np.empty(0, dtype=np.int64)
                continue
            idx = np.asarray(idx, dtype=np.int64)
            yield idx[np.abs(self.minutes[idx] - anchor_min[i]) <= window_min]

    def any_within(self, lat, lon, minutes, radius_km: float, window_min: int) -> np.ndarray:
        """Boolean mask: does each query point have at least one match?"""
        out = np.zeros(len(np.asarray(lat)), dtype=int)
        for i, idx in enumerate(self.query(lat, lon, minutes, radius_km, window_min)):
            out[i] = int(idx.size > 0)
        return out


def as_frame(lat, lon, minutes) -> pd.DataFrame:   # pragma: no cover - convenience
    return pd.DataFrame({"latitude": lat, "longitude": lon, "minutes": minutes})
