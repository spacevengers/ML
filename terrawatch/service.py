"""HTTP service implementing the Go backend's remote harmonizer contract."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .predict import predict_bundle

EARTH_RADIUS_KM = 6371.0088
MAX_BODY_BYTES = 128 * 1024 * 1024


# --------------------------------------------------------------------------- #
# geometry / time
# --------------------------------------------------------------------------- #

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def parse_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("acquired_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid acquired_at {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #

def _cell_size(max_distance_km: float, max_abs_latitude: float) -> tuple[float, float]:
    lat_degrees = max_distance_km / 110.574
    cos_lat = max(0.05, math.cos(math.radians(max_abs_latitude)))
    lon_degrees = max_distance_km / (111.320 * cos_lat)
    return lat_degrees, lon_degrees


def _prepare(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"observation {index} must be an object")
        try:
            observation_id = int(observation["observation_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"observation {index} has an invalid observation_id") from exc
        if observation_id in seen:
            raise ValueError(f"duplicate observation_id {observation_id}")
        seen.add(observation_id)

        sensor = str(observation.get("sensor") or "").upper()
        if sensor not in ("MODIS", "VIIRS"):
            raise ValueError(f"observation {observation_id} has sensor {sensor!r}")

        try:
            latitude = float(observation["latitude"])
            longitude = float(observation["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"observation {observation_id} has invalid latitude/longitude"
            ) from exc
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(f"observation {observation_id} is outside valid coordinates")

        prepared.append({
            "id": observation_id,
            "uid": str(observation.get("observation_uid") or observation_id),
            "sensor": sensor,
            "satellite": observation.get("satellite") or "unknown",
            "lat": latitude,
            "lon": longitude,
            "time": parse_time(observation.get("acquired_at")),
            "frp": _number(observation.get("frp")),
            "confidence_class": observation.get("confidence_class") or "unknown",
        })
    return prepared


def _cluster_uid(run_uid: str, members: list[dict[str, Any]]) -> str:
    canonical = run_uid + "|" + "|".join(
        sorted(member["uid"] for member in members)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_cluster(run_uid: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(members, key=lambda member: member["id"])
    count = len(members)
    centroid_lat = sum(member["lat"] for member in members) / count
    centroid_lon = sum(member["lon"] for member in members) / count
    start = min(member["time"] for member in members)
    end = max(member["time"] for member in members)

    modis_count = sum(member["sensor"] == "MODIS" for member in members)
    viirs_count = count - modis_count
    frp_values = [member["frp"] for member in members if member["frp"] is not None]
    satellites = sorted({member["satellite"] for member in members})

    confidence: dict[str, int] = defaultdict(int)
    for member in members:
        confidence[str(member["confidence_class"])] += 1

    radius = max(
        haversine_km(centroid_lat, centroid_lon, member["lat"], member["lon"])
        for member in members
    )

    total_frp = round(sum(frp_values), 4) if frp_values else None
    mean_frp = round(sum(frp_values) / len(frp_values), 4) if frp_values else None

    return {
        "cluster_uid": _cluster_uid(run_uid, members),
        "members": [
            {"observation_id": member["id"], "role": member["sensor"].lower()}
            for member in members
        ],
        "centroid_latitude": centroid_lat,
        "centroid_longitude": centroid_lon,
        "start_time": iso_time(start),
        "end_time": iso_time(end),
        "observation_count": count,
        "modis_count": modis_count,
        "viirs_count": viirs_count,
        "satellites": satellites,
        "total_frp": total_frp,
        "mean_frp": mean_frp,
        "confidence_summary": dict(confidence),
        "uncertainty": {
            "spatial_radius_km": round(radius, 4),
            "time_span_minutes": round((end - start).total_seconds() / 60, 3),
        },
    }


def cluster_observations(
    observations: list[dict[str, Any]],
    *,
    run_uid: str,
    max_distance_km: float,
    max_time_minutes: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster observations with the same one-directional rule as the backend."""
    if max_distance_km <= 0 or max_time_minutes <= 0:
        raise ValueError("max_distance_km and max_time_minutes must be positive")

    prepared = _prepare(observations)
    modis = [item for item in prepared if item["sensor"] == "MODIS"]
    viirs = [item for item in prepared if item["sensor"] == "VIIRS"]

    if not modis or not viirs:
        return [_build_cluster(run_uid, [item]) for item in prepared], []

    max_abs_latitude = max(abs(item["lat"]) for item in prepared)
    cell_lat, cell_lon = _cell_size(max_distance_km, max_abs_latitude)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    modis_position: dict[int, int] = {}
    for index, item in enumerate(modis):
        key = (math.floor(item["lat"] / cell_lat), math.floor(item["lon"] / cell_lon))
        grid[key].append(index)
        modis_position[item["id"]] = index

    groups: list[list[dict[str, Any]]] = [[item] for item in modis]
    matches: list[dict[str, Any]] = []

    for target in viirs:
        key_lat = math.floor(target["lat"] / cell_lat)
        key_lon = math.floor(target["lon"] / cell_lon)
        best: dict[str, Any] | None = None
        best_distance = math.inf
        best_delta = math.inf
        for delta_lat in (-1, 0, 1):
            for delta_lon in (-1, 0, 1):
                for index in grid.get((key_lat + delta_lat, key_lon + delta_lon), []):
                    candidate = modis[index]
                    delta_minutes = abs(
                        (target["time"] - candidate["time"]).total_seconds() / 60
                    )
                    if delta_minutes > max_time_minutes:
                        continue
                    distance = haversine_km(
                        target["lat"], target["lon"],
                        candidate["lat"], candidate["lon"],
                    )
                    if distance > max_distance_km:
                        continue
                    if distance < best_distance or (
                        distance == best_distance and delta_minutes < best_delta
                    ):
                        best = candidate
                        best_distance = distance
                        best_delta = delta_minutes
        if best is None:
            groups.append([target])
            continue
        groups[modis_position[best["id"]]].append(target)
        matches.append({
            "modis_observation_id": best["id"],
            "viirs_observation_id": target["id"],
            "distance_km": round(best_distance, 4),
            "time_delta_minutes": round(best_delta, 3),
        })

    clusters = [_build_cluster(run_uid, members) for members in groups]
    clusters.sort(key=lambda cluster: (cluster["start_time"], cluster["cluster_uid"]))
    return clusters, matches


# --------------------------------------------------------------------------- #
# harmonize
# --------------------------------------------------------------------------- #

def harmonize(payload: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Handle one remote harmonizer request and attach model predictions."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty array")

    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    max_distance = _number(params.get("max_distance_km")) or 1.5
    max_time = _number(params.get("max_time_minutes")) or 30.0
    run_uid = str(payload.get("run_uid") or "run")

    prepared = _prepare(observations)
    modis_count = sum(item["sensor"] == "MODIS" for item in prepared)
    viirs_count = len(prepared) - modis_count

    clusters, matches = cluster_observations(
        observations,
        run_uid=run_uid,
        max_distance_km=max_distance,
        max_time_minutes=max_time,
    )

    scoring_metrics: dict[str, Any] = {}
    scores: dict[str, float] = {}
    try:
        result = predict_bundle(
            {
                "observations": observations,
                "clusters": [
                    {"cluster_uid": cluster["cluster_uid"], "members": cluster["members"]}
                    for cluster in clusters
                ],
            },
            artifact,
            aggregation="max",
        )
        scores = {
            item["cluster_uid"]: item["p_modis_detect"]
            for item in result["predictions"]
        }
        scoring_metrics = {
            "scored_clusters": len(scores),
            "scoring": result.get("metrics", {}),
        }
    except Exception as exc:  # keep clustering usable even if scoring fails
        scoring_metrics = {"scoring_error": str(exc)}

    for cluster in clusters:
        score = scores.get(cluster["cluster_uid"])
        if score is not None:
            cluster["p_modis_detect"] = score
            cluster["prediction_model"] = artifact.get("version", "unknown")

    return {
        "clusters": clusters,
        "matches": matches,
        "metrics": {
            "method": "lightgbm-remote",
            "model_version": artifact.get("version"),
            "feature_set": artifact.get("feature_set"),
            "input_count": len(prepared),
            "modis_count": modis_count,
            "viirs_count": viirs_count,
            "cluster_count": len(clusters),
            "match_count": len(matches),
            **scoring_metrics,
        },
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class HarmonizerHandler(BaseHTTPRequestHandler):
    artifact: dict[str, Any] = {}
    verbose: bool = False

    server_version = "TerraWatchML/0.1"
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path in ("/health", "/ready"):
            self._send(200, {
                "status": "ok",
                "model": self.artifact.get("version"),
                "feature_set": self.artifact.get("feature_set"),
                "model_type": self.artifact.get("model_type"),
            })
            return
        self._send(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path != "/harmonize":
            self._send(404, {"detail": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"detail": "invalid Content-Length"})
            return
        if length <= 0:
            self._send(400, {"detail": "empty request body"})
            return
        if length > MAX_BODY_BYTES:
            self._send(413, {"detail": "request body too large"})
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"detail": f"invalid JSON: {exc}"})
            return

        try:
            result = harmonize(payload, self.artifact)
        except ValueError as exc:
            self._send(422, {"detail": str(exc)})
            return
        except Exception as exc:
            if self.verbose:
                print(f"harmonize failed: {exc}", file=sys.stderr)
            self._send(500, {"detail": "internal error"})
            return

        self._send(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        if self.verbose:
            super().log_message(format, *args)


def make_handler(artifact: dict[str, Any], *, verbose: bool = False):
    class BoundHandler(HarmonizerHandler):
        pass

    BoundHandler.artifact = artifact
    BoundHandler.verbose = verbose
    return BoundHandler


def run_server(
    artifact: dict[str, Any],
    *,
    host: str = "0.0.0.0",
    port: int = 9000,
    verbose: bool = False,
) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(artifact, verbose=verbose))
    print(
        f"TerraWatch ML service listening on http://{host}:{port} "
        f"(model {artifact.get('version')})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
