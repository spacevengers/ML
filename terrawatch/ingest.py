"""FIRMS acquisition: parallel, chunked, cached as Parquet."""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config
from .config import Settings

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def chunk_starts(start: date, end: date, step: int = 5):
    """Yield chunk start dates covering [start, end)."""
    d = start
    while d < end:
        yield d
        d += timedelta(days=step)


@dataclass
class FirmsClient:
    """Download active-fire detections and cache them on disk.

    Cache key is (source product, chunk start). Re-running a range costs nothing.
    """

    settings: Settings = config.DEFAULT
    raw_dir: Path = config.RAW_DIR
    key: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            self.key = config.firms_key()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # single request
    # ------------------------------------------------------------------ #
    def _url(self, source: str, start: date, days: int) -> str:
        return (f"{BASE}/{self.key}/{source}/{self.settings.bbox_str}"
                f"/{days}/{start.isoformat()}")

    def _cache_path(self, source: str, start: date) -> Path:
        return self.raw_dir / f"{source}_{start.isoformat()}.parquet"

    def _download(self, url: str, timeout: int = 90) -> pd.DataFrame:
        last: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    body = resp.read().decode()
                if body.lstrip().startswith("Invalid"):
                    raise ValueError(body.strip()[:200])
                return pd.read_csv(io.StringIO(body))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"download failed after 3 attempts: {last}") from last

    # ------------------------------------------------------------------ #
    # chunk
    # ------------------------------------------------------------------ #
    def fetch_chunk(self, source: str, start: date, days: int | None = None) -> pd.DataFrame:
        days = days or self.settings.chunk_days
        path = self._cache_path(source, start)
        if path.exists() and path.stat().st_size > 0:
            return pd.read_parquet(path)

        df = self._download(self._url(source, start, days))
        df = df.drop_duplicates().reset_index(drop=True)
        df["source_product"] = source
        df["retrieved_at"] = datetime.now().isoformat(timespec="seconds")
        df.to_parquet(path, index=False)
        return df

    # ------------------------------------------------------------------ #
    # range
    # ------------------------------------------------------------------ #
    def fetch(self, start: date, end: date, *, progress: bool = True) -> dict[str, pd.DataFrame]:
        """Fetch every source over [start, end), in parallel."""
        jobs = [
            (name, source, chunk)
            for name, source in self.settings.sources.items()
            for chunk in chunk_starts(start, end, self.settings.chunk_days)
        ]

        frames: dict[tuple[str, date], pd.DataFrame] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=self.settings.fetch_workers) as pool:
            futures = {
                pool.submit(self.fetch_chunk, source, chunk): (name, chunk)
                for name, source, chunk in jobs
            }
            for fut in as_completed(futures):
                name, chunk = futures[fut]
                frames[(name, chunk)] = fut.result()
                done += 1
                if progress:
                    print(f"  [{done}/{len(jobs)}] {name} {chunk}", flush=True)

        out: dict[str, pd.DataFrame] = {}
        for name in self.settings.sources:
            parts = [f for (n, _), f in sorted(frames.items(), key=lambda kv: kv[0][1])
                     if n == name and len(f)]
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if len(df):
                df = df.drop_duplicates().reset_index(drop=True)
            out[name] = df
        return out

    def load(self, start: date, end: date) -> dict[str, pd.DataFrame]:
        """Read cached chunks only; never hits the network."""
        out: dict[str, pd.DataFrame] = {}
        for name, source in self.settings.sources.items():
            parts = [
                pd.read_parquet(p)
                for c in chunk_starts(start, end, self.settings.chunk_days)
                if (p := self._cache_path(source, c)).exists()
            ]
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            out[name] = df.drop_duplicates().reset_index(drop=True) if len(df) else df
        return out
