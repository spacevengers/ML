"""
FIRMS client: download MODIS/VIIRS active-fire detections and cache them as CSV.

The area API returns at most DAYS=5 days per call, so ranges are fetched in
5-day chunks and cached on disk. Re-running is free: existing chunks are reused.

Usage:
    python3 firms_client.py 2023-02-10 2023-02-15
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

KEY = os.environ.get("FIRMS_KEY")
if not KEY:
    raise SystemExit(
        "FIRMS_KEY is not set.\n"
        "  cp ML/.env.example ML/.env   # then paste your key into it\n"
        "  free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/"
    )

BBOX = "33.9,-4.7,41.9,5.2"  # west,south,east,north -> Kenya

# _SP = standard processing (archive). Use _NRT only for recent data.
SOURCES = {
    "modis": "MODIS_SP",
    "viirs_snpp": "VIIRS_SNPP_SP",
}

DAYS = 5  # API maximum


def _chunks(start: date, end: date, step: int = DAYS):
    d = start
    while d < end:
        yield d
        d += timedelta(days=step)


def _path(source: str, start: date) -> Path:
    return RAW / f"{source}_{start.isoformat()}.csv"


def fetch_chunk(source: str, start: date, days: int = DAYS) -> pd.DataFrame:
    """Fetch (or reuse) a single <=5 day chunk for one product."""
    p = _path(source, start)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{KEY}/{source}/{BBOX}/{days}/{start.isoformat()}"
    )
    df = pd.read_csv(url)
    df.to_csv(p, index=False)
    time.sleep(0.5)  # be polite to the API
    return df


def load_raw(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (modis, viirs) covering [start, end), concatenated and de-duplicated."""
    out = {}
    for name, source in SOURCES.items():
        frames = [fetch_chunk(source, d) for d in _chunks(start, end)]
        frames = [f for f in frames if len(f)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            df = df.drop_duplicates().reset_index(drop=True)
        out[name] = df
        print(f"  {name:10s} {len(df):7,d} rows  {source}")
    return out["modis"], out["viirs_snpp"]


if __name__ == "__main__":
    import sys

    s = datetime.fromisoformat(sys.argv[1]).date()
    e = datetime.fromisoformat(sys.argv[2]).date()
    print(f"Fetching {s} .. {e} (exclusive)")
    load_raw(s, e)
