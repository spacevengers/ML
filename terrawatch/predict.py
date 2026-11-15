"""Prediction: turn a backend bundle into cluster-level ML output."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .features import add_features
from .labels import build_viirs_features

AGGREGATIONS = ("max", "mean")


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _first(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and value != "":
            return value
    return None


def _confidence_letter(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("l", "low"):
            return "l"
        if lowered in ("n", "nominal"):
            return "n"
        if lowered in ("h", "high"):
            return "h"
        try:
            value = float(lowered)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 30:
        return "l"
    if number < 80:
        return "n"
    return "h"


def _acquisition_fields(observation: dict[str, Any], index: int) -> tuple[str, int]:
    date = observation.get("acq_date")
    time = observation.get("acq_time")
    if date is not None and time is not None:
        try:
            return str(date), int(str(time).zfill(4))
        except (TypeError, ValueError):
            pass

    acquired = observation.get("acquired_at") or observation.get("acq_datetime")
    if acquired:
        timestamp = pd.to_datetime(acquired, utc=True, errors="coerce")
        if not pd.isna(timestamp):
            return timestamp.strftime("%Y-%m-%d"), int(timestamp.strftime("%H%M"))

    raise ValueError(
        f"observation {observation.get('observation_id', index)!r} has no usable acquisition time"
    )


def normalize_observations(observations: list[dict[str, Any]]) -> pd.DataFrame:
    """Map backend observation JSON to the FIRMS-like frame used in training."""
    rows: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        date, time = _acquisition_fields(observation, index)
        confidence = _first(observation, ("confidence_class", "confidence"))
        rows.append({
            "observation_id": observation.get("observation_id", observation.get("id", index)),
            "sensor": str(observation.get("sensor") or "").upper(),
            "satellite": observation.get("satellite") or "unknown",
            "source_product": observation.get("source_product") or observation.get("source") or "VIIRS",
            "acq_date": date,
            "acq_time": time,
            "latitude": _to_float(observation.get("latitude")),
            "longitude": _to_float(observation.get("longitude")),
            "frp": _to_float(observation.get("frp")),
            "bright_ti4": _to_float(
                _first(observation, ("bright_ti4", "brightness"))
            ),
            "bright_ti5": _to_float(
                _first(observation, ("bright_ti5", "brightness_aux", "bright_t31"))
            ),
            "scan": _to_float(observation.get("scan")),
            "track": _to_float(observation.get("track")),
            "confidence": _confidence_letter(confidence),
            "daynight": str(observation.get("daynight") or "").upper(),
        })
    return pd.DataFrame(rows)


def settings_from_artifact(artifact: dict[str, Any]) -> Settings:
    raw = dict(artifact.get("settings") or {})
    allowed = {field.name for field in dataclasses.fields(Settings)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    if kwargs.get("bbox") is not None:
        kwargs["bbox"] = tuple(kwargs["bbox"])
    return Settings(**kwargs)


def _cluster_member_ids(cluster: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for member in cluster.get("members") or []:
        if isinstance(member, dict):
            role = str(member.get("role") or "").lower()
            if role and role != "viirs":
                continue
            observation_id = member.get("observation_id")
        else:
            observation_id = member
        if observation_id is not None:
            ids.append(str(observation_id))
    return ids


def _aggregate(values: list[float], aggregation: str) -> float:
    if aggregation == "mean":
        return float(np.mean(values))
    return float(np.max(values))


def predict_bundle(
    bundle: dict[str, Any],
    artifact: dict[str, Any],
    *,
    aggregation: str = "max",
) -> dict[str, Any]:
    """Score a backend bundle and aggregate to one prediction per cluster.

    ``bundle`` must contain:
      - ``observations``: every VIIRS observation in the run's date/area, not
        just cluster members, so footprint features are correct;
      - ``clusters``: cluster_uid plus members with observation_id/role.

    The model is trained on VIIRS rows, so MODIS-only clusters are skipped.
    ``frp_calibrated`` is intentionally not emitted: the current model does not
    predict FRP.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")

    observations = bundle.get("observations") or []
    if not observations:
        raise ValueError("bundle has no observations")
    clusters = bundle.get("clusters") or []
    if not clusters:
        raise ValueError("bundle has no clusters")

    frame = normalize_observations(observations)
    if frame.empty:
        raise ValueError("bundle observations could not be parsed")

    viirs = frame[frame["sensor"] == "VIIRS"].reset_index(drop=True)
    if viirs.empty:
        raise ValueError("bundle contains no VIIRS observations")

    settings = settings_from_artifact(artifact)
    features, _ = build_viirs_features(viirs, settings, id_column="observation_id")
    feature_names = list(artifact["features"])
    enriched = add_features(features)
    missing = [name for name in feature_names if name not in enriched.columns]
    if missing:
        raise ValueError(f"feature frame is missing model inputs: {', '.join(missing)}")

    model_frame = enriched[feature_names]
    probabilities = artifact["model"].predict_proba(model_frame)
    scores = np.asarray(probabilities)[:, -1]
    scores_by_id = {
        str(observation_id): float(np.clip(score, 0.0, 1.0))
        for observation_id, score in zip(features["observation_id"], scores)
    }

    predictions: list[dict[str, Any]] = []
    skipped_no_viirs = 0
    skipped_no_score = 0
    for cluster in clusters:
        cluster_uid = cluster.get("cluster_uid")
        if not cluster_uid:
            skipped_no_score += 1
            continue
        member_ids = _cluster_member_ids(cluster)
        values = [scores_by_id[member] for member in member_ids if member in scores_by_id]
        if not values:
            skipped_no_viirs += 1
            continue
        predictions.append({
            "cluster_uid": str(cluster_uid),
            "p_modis_detect": round(_aggregate(values, aggregation), 6),
        })

    if not predictions:
        raise ValueError("no cluster had a scorable VIIRS member")

    return {
        "model": artifact.get("version", "unknown"),
        "feature_set": artifact.get("feature_set"),
        "predictions": predictions,
        "metrics": {
            "observations_in_bundle": int(len(frame)),
            "viirs_observations_scored": int(len(features)),
            "clusters_total": len(clusters),
            "clusters_scored": len(predictions),
            "clusters_skipped_no_viirs": skipped_no_viirs,
            "clusters_skipped_other": skipped_no_score,
            "aggregation": aggregation,
            "model_type": artifact.get("model_type"),
        },
    }


def read_bundle(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_predictions(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
