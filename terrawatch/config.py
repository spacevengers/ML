"""Central configuration: paths, thresholds, and environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent   # ML/
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
LABELS_DIR = DATA / "labels"
MODELS_DIR = DATA / "models"

load_dotenv(ROOT / ".env")


def firms_key() -> str:
    """Return the FIRMS MAP_KEY or exit with a helpful message."""
    key = os.environ.get("FIRMS_KEY")
    if not key:
        raise SystemExit(
            "FIRMS_KEY is not set.\n"
            "  cp ML/.env.example ML/.env   # then paste your key into it\n"
            "  free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )
    return key


@dataclass(frozen=True)
class Settings:
    """Everything tunable in one place."""

    # --- study area (west, south, east, north) ---
    bbox: tuple[float, float, float, float] = (33.9, -4.7, 41.9, 5.2)   # Kenya

    # --- FIRMS products ---
    sources: dict[str, str] = field(
        default_factory=lambda: {
            "modis": "MODIS_SP",
            "viirs": "VIIRS_SNPP_SP",
        }
    )
    chunk_days: int = 5          # API maximum
    fetch_workers: int = 4       # parallel requests

    # --- labelling ---
    match_radius_km: float = 1.5
    match_window_min: int = 30
    neigh_radius_km: float = 1.5
    neigh_window_min: int = 5
    max_scan_km: float | None = None   # filter VIIRS swath-edge detections

    # --- evaluation ---
    block_deg: float = 1.0
    seed: int = 42

    @property
    def bbox_str(self) -> str:
        return ",".join(f"{v:g}" for v in self.bbox)


DEFAULT = Settings()


def ensure_dirs() -> None:
    for d in (RAW_DIR, LABELS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
