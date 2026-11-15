import dataclasses

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from terrawatch.artifacts import latest_artifact, load_artifact, save_artifact
from terrawatch.config import Settings
from terrawatch.features import FEATURE_SETS
from terrawatch.labels import FEATURE_COLUMNS, build_label_table, build_viirs_features
from terrawatch.predict import normalize_observations, predict_bundle


def _viirs_row(lat=0.0, lon=0.0, time=1000, frp=10.0, conf="n"):
    return {
        "acq_date": "2023-02-10",
        "acq_time": time,
        "latitude": lat,
        "longitude": lon,
        "frp": frp,
        "bright_ti4": 340.0,
        "bright_ti5": 295.0,
        "scan": 0.4,
        "track": 0.4,
        "confidence": conf,
        "daynight": "D",
        "satellite": "N",
        "source_product": "VIIRS_SNPP_SP",
    }


def _modis_row(lat=0.0, lon=0.0, time=1000):
    return {
        "acq_date": "2023-02-10",
        "acq_time": time,
        "latitude": lat,
        "longitude": lon,
        "frp": 30.0,
        "brightness": 330.0,
        "bright_t31": 300.0,
        "scan": 1.0,
        "track": 1.0,
        "confidence": 80,
        "daynight": "D",
        "satellite": "Aqua",
        "source_product": "MODIS_SP",
    }


class FixedProbabilityModel:
    """Small test double with sklearn's predict_proba shape."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        values = self.probabilities[: len(frame)]
        return np.column_stack([1.0 - values, values])


def test_serving_features_match_training_features():
    viirs = pd.DataFrame([
        _viirs_row(0.0, 0.0, 1000, 10.0),
        _viirs_row(0.001, 0.0, 1002, 20.0),
        _viirs_row(0.0, 0.001, 1003, 30.0),
    ])
    modis = pd.DataFrame([_modis_row()])

    training = build_label_table(modis, viirs, Settings())
    serving, _ = build_viirs_features(viirs, Settings())

    pd.testing.assert_frame_equal(
        training[FEATURE_COLUMNS].reset_index(drop=True),
        serving[FEATURE_COLUMNS].reset_index(drop=True),
    )


def test_normalize_backend_observation_fields():
    frame = normalize_observations([{
        "observation_id": 42,
        "sensor": "VIIRS",
        "satellite": "Suomi NPP",
        "acquired_at": "2024-01-01T10:32:00Z",
        "frp": 4.2,
        "brightness": 320.1,
        "brightness_aux": 290.4,
        "scan": 0.4,
        "track": 0.5,
        "confidence_class": "nominal",
        "daynight": "N",
        "source_product": "VIIRS_SNPP_SP",
    }])

    row = frame.iloc[0]
    assert row["observation_id"] == 42
    assert row["acq_date"] == "2024-01-01"
    assert row["acq_time"] == 1032
    assert row["bright_ti4"] == 320.1
    assert row["bright_ti5"] == 290.4
    assert row["confidence"] == "n"
    assert row["daynight"] == "N"


def test_predict_bundle_aggregates_per_cluster_and_skips_modis_only():
    observations = [
        {
            "observation_id": index,
            "sensor": "VIIRS",
            "satellite": "Suomi NPP",
            "acquired_at": "2024-01-01T10:32:00Z",
            "latitude": 0.0,
            "longitude": 0.0,
            "frp": 10.0 + index,
            "brightness": 340.0,
            "brightness_aux": 295.0,
            "scan": 0.4,
            "track": 0.4,
            "confidence_class": "nominal",
            "daynight": "D",
        }
        for index in (1, 2, 3)
    ]
    clusters = [
        {
            "cluster_uid": "cluster-a",
            "members": [
                {"observation_id": 1, "role": "viirs"},
                {"observation_id": 2, "role": "viirs"},
            ],
        },
        {
            "cluster_uid": "cluster-b",
            "members": [{"observation_id": 3, "role": "viirs"}],
        },
        {
            "cluster_uid": "cluster-c",
            "members": [{"observation_id": 4, "role": "modis"}],
        },
    ]
    artifact = {
        "version": "test-model",
        "model_type": "FixedProbabilityModel",
        "feature_set": "physics+geom+night",
        "features": FEATURE_SETS["physics+geom+night"],
        "settings": dataclasses.asdict(Settings()),
        "model": FixedProbabilityModel([0.2, 0.8, 0.5]),
    }

    result = predict_bundle(
        {"observations": observations, "clusters": clusters}, artifact
    )

    predictions = {
        item["cluster_uid"]: item["p_modis_detect"]
        for item in result["predictions"]
    }
    assert predictions == {"cluster-a": 0.8, "cluster-b": 0.5}
    assert result["metrics"]["clusters_skipped_no_viirs"] == 1


def test_artifact_roundtrip(tmp_path):
    model = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    features = ["log_frp"]

    path = save_artifact(
        model,
        features,
        "physics",
        Settings(),
        version="test-artifact",
        directory=tmp_path,
        metrics={"grouped_cv": {"lightgbm": {"PR-AUC": 0.5}}},
    )

    loaded = load_artifact(path)
    assert loaded["version"] == "test-artifact"
    assert loaded["features"] == features
    assert loaded["feature_set"] == "physics"
    assert latest_artifact(tmp_path) == path
