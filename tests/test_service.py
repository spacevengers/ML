import dataclasses
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from terrawatch.config import Settings
from terrawatch.features import FEATURE_SETS
from terrawatch.service import harmonize, make_handler


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        values = self.probabilities[: len(frame)]
        return np.column_stack([1.0 - values, values])


def _observation(observation_id, sensor, lat, lon, time, frp, satellite):
    return {
        "observation_id": observation_id,
        "observation_uid": f"uid-{observation_id}",
        "sensor": sensor,
        "satellite": satellite,
        "latitude": lat,
        "longitude": lon,
        "acquired_at": time,
        "frp": frp,
        "brightness": 340.0,
        "brightness_aux": 295.0,
        "scan": 0.4,
        "track": 0.4,
        "confidence_class": "nominal",
        "daynight": "D",
    }


def _artifact():
    return {
        "version": "test-service",
        "model_type": "FixedProbabilityModel",
        "feature_set": "physics+geom+night",
        "features": FEATURE_SETS["physics+geom+night"],
        "settings": dataclasses.asdict(Settings()),
        "model": FixedProbabilityModel([0.9, 0.1]),
    }


def _request():
    return {
        "run_uid": "run-test",
        "method": "remote",
        "params": {"max_distance_km": 1.5, "max_time_minutes": 30},
        "observations": [
            _observation(1, "MODIS", 0.0, 0.0, "2024-01-01T10:30:00Z", 12.0, "Aqua"),
            _observation(2, "VIIRS", 0.001, 0.0, "2024-01-01T10:31:00Z", 10.0, "Suomi NPP"),
            _observation(3, "VIIRS", 1.0, 1.0, "2024-01-01T10:31:00Z", 5.0, "Suomi NPP"),
        ],
    }


def test_harmonize_clusters_and_scores():
    result = harmonize(_request(), _artifact())

    assert len(result["clusters"]) == 2
    assert len(result["matches"]) == 1
    assert result["metrics"]["input_count"] == 3
    assert result["metrics"]["scored_clusters"] == 2

    by_counts = {
        (cluster["modis_count"], cluster["viirs_count"]): cluster
        for cluster in result["clusters"]
    }
    mixed = by_counts[(1, 1)]
    singleton = by_counts[(0, 1)]

    assert mixed["p_modis_detect"] == 0.9
    assert mixed["prediction_model"] == "test-service"
    assert singleton["p_modis_detect"] == 0.1
    assert mixed["members"] == [
        {"observation_id": 1, "role": "modis"},
        {"observation_id": 2, "role": "viirs"},
    ]
    assert len(mixed["cluster_uid"]) == 64


def test_http_health_and_harmonize():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_artifact()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            health = json.loads(response.read())
        assert health["status"] == "ok"
        assert health["model"] == "test-service"

        body = json.dumps(_request()).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/harmonize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        assert len(payload["clusters"]) == 2
        assert payload["metrics"]["model_version"] == "test-service"
    finally:
        server.shutdown()
        server.server_close()
