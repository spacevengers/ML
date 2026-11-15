"""Read a harmonization run from the Go backend and import predictions."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = 60,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Admin-Token"] = token
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    return json.loads(body) if body else None


def _url(base: str, path: str, params: dict[str, Any] | None = None) -> str:
    url = f"{base.rstrip('/')}{path}"
    if params:
        clean = {key: value for key, value in params.items() if value not in (None, "")}
        if clean:
            url = f"{url}?{urllib.parse.urlencode(clean)}"
    return url


def latest_completed_run(base: str) -> dict[str, Any]:
    payload = _request(
        _url(base, "/api/harmonization-runs", {"status": "completed", "limit": 1})
    )
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not items:
        raise RuntimeError("no completed harmonization run found")
    return items[0]


def get_run(base: str, run_id: int) -> dict[str, Any]:
    payload = _request(_url(base, f"/api/harmonization-runs/{run_id}"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"run {run_id} did not return an object")
    return payload


def _bbox_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        try:
            return ",".join(
                str(value[key]) for key in ("west", "south", "east", "north")
            )
        except KeyError as exc:
            raise RuntimeError(f"run bbox is missing {exc.args[0]!r}") from exc
    raise RuntimeError(f"unsupported bbox value: {value!r}")


def run_filters(
    run: dict[str, Any],
    *,
    start: str | None = None,
    end: str | None = None,
    bbox: str | None = None,
) -> dict[str, str | None]:
    """Extract the run's observation filter, allowing CLI overrides."""
    filters = run.get("filters") or {}
    if not isinstance(filters, dict):
        filters = {}
    return {
        "start": start or filters.get("start"),
        "end": end or filters.get("end"),
        "bbox": bbox or _bbox_string(filters.get("bbox")),
    }


def fetch_observations(
    base: str,
    *,
    start: str | None = None,
    end: str | None = None,
    bbox: str | None = None,
    page_size: int = 5000,
) -> list[dict[str, Any]]:
    """Fetch every VIIRS observation matching the run filter."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = _request(
            _url(
                base,
                "/api/observations",
                {
                    "sensor": "VIIRS",
                    "start": start,
                    "end": end,
                    "bbox": bbox,
                    "limit": page_size,
                    "offset": offset,
                    "sort_by": "acquired_at",
                    "order": "asc",
                },
            )
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        rows.extend(items)
        total = int(payload.get("total", len(rows))) if isinstance(payload, dict) else len(rows)
        if not items or len(rows) >= total:
            break
        offset += len(items)
    return rows


def fetch_clusters(
    base: str, run_id: int, *, page_size: int = 500
) -> list[dict[str, Any]]:
    """Fetch cluster UIDs and member IDs for one run."""
    clusters: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = _request(
            _url(
                base,
                "/api/harmonized-observations",
                {
                    "run_id": run_id,
                    "include_members": "true",
                    "limit": page_size,
                    "offset": offset,
                },
            )
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for cluster in items:
            members = []
            for member in cluster.get("members") or []:
                if not isinstance(member, dict):
                    continue
                members.append({
                    "observation_id": member.get("observation_id"),
                    "role": member.get("role"),
                })
            clusters.append({
                "cluster_uid": cluster.get("cluster_uid"),
                "members": members,
            })
        total = int(payload.get("total", len(clusters))) if isinstance(payload, dict) else len(clusters)
        if not items or offset + len(items) >= total:
            break
        offset += len(items)
    return clusters


def build_bundle(
    base: str,
    *,
    run_id: int | None = None,
    start: str | None = None,
    end: str | None = None,
    bbox: str | None = None,
) -> dict[str, Any]:
    """Build the JSON bundle consumed by ``terrawatch predict``."""
    run = get_run(base, run_id) if run_id else latest_completed_run(base)
    filters = run_filters(run, start=start, end=end, bbox=bbox)
    clusters = fetch_clusters(base, int(run["id"]))
    observations = fetch_observations(base, **filters)
    return {
        "run_id": int(run["id"]),
        "run_uid": run.get("run_uid"),
        "rule": run.get("params") or {},
        "clusters": clusters,
        "observations": observations,
    }


def import_predictions(
    base: str,
    run_id: int,
    result: dict[str, Any],
    *,
    token: str | None = None,
) -> Any:
    """POST a prediction envelope to the backend import endpoint."""
    return _request(
        _url(base, f"/api/harmonization-runs/{run_id}/predictions"),
        method="POST",
        token=token,
        payload={
            "model": result.get("model", "external"),
            "predictions": result.get("predictions", []),
        },
    )
