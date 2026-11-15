"""Versioned model artifacts shared by training and prediction."""

from __future__ import annotations

import dataclasses
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from . import __version__
from .config import MODELS_DIR, Settings

ARTIFACT_SCHEMA = 1
REQUIRED_KEYS = ("model", "features", "feature_set", "settings")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_version(prefix: str = "lightgbm") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def artifact_path(version: str, directory: Path = MODELS_DIR) -> Path:
    return directory / f"{version}.joblib"


def metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_artifact(
    model: Any,
    features: list[str],
    feature_set: str,
    settings: Settings,
    *,
    version: str | None = None,
    training: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    directory: Path = MODELS_DIR,
) -> Path:
    """Persist a fitted estimator and the contract needed to serve it."""
    version = version or default_version()
    directory.mkdir(parents=True, exist_ok=True)
    path = artifact_path(version, directory)

    payload = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "version": version,
        "created_at": utc_now(),
        "model_type": type(model).__name__,
        "model": model,
        "features": list(features),
        "feature_set": feature_set,
        "settings": dataclasses.asdict(settings),
        "training": training or {},
        "metrics": metrics or {},
        "terrawatch_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    joblib.dump(payload, path)

    metadata = {key: value for key, value in payload.items() if key != "model"}
    metadata["artifact_file"] = path.name
    metadata_path(path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate an artifact written by :func:`save_artifact`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a TerraWatch artifact")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing artifact keys: {', '.join(missing)}")
    if payload.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError(
            f"{path} uses artifact schema {payload.get('artifact_schema')!r}; "
            f"this code expects {ARTIFACT_SCHEMA}"
        )
    return payload


def latest_artifact(directory: Path = MODELS_DIR) -> Path | None:
    """Return the newest ``.joblib`` artifact in a directory, if any."""
    if not directory.exists():
        return None
    candidates = sorted(
        directory.glob("*.joblib"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None
