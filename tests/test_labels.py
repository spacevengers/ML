import dataclasses

import pandas as pd

from terrawatch.config import Settings
from terrawatch.labels import build_label_table


def _viirs(lat, lon, time, frp=10.0, conf="n"):
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


def _modis(lat, lon, time):
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


def _build(viirs_rows, modis_rows, **settings):
    cfg = dataclasses.replace(Settings(), **settings) if settings else Settings()
    return build_label_table(
        pd.DataFrame(modis_rows), pd.DataFrame(viirs_rows), cfg
    )


def test_one_row_per_viirs_detection():
    viirs = [_viirs(0.0, 0.0, 1000), _viirs(5.0, 5.0, 1000)]
    modis = [_modis(0.0, 0.0, 1000)]
    assert len(_build(viirs, modis)) == 2


def test_nearby_detection_is_labelled_positive():
    viirs = [_viirs(0.0, 0.0, 1000)]
    out = _build(viirs, [_modis(0.001, 0.0, 1010)])
    assert out["modis_detected"].iloc[0] == 1


def test_far_detection_is_labelled_negative():
    viirs = [_viirs(0.0, 0.0, 1000)]
    out = _build(viirs, [_modis(0.5, 0.5, 1000)])   # ~78 km away
    assert out["modis_detected"].iloc[0] == 0


def test_detection_outside_time_window_is_negative():
    viirs = [_viirs(0.0, 0.0, 1000)]
    out = _build(viirs, [_modis(0.0, 0.0, 1300)])   # 5 hours later
    assert out["modis_detected"].iloc[0] == 0


def test_neighbours_are_aggregated_into_footprint_features():
    # three co-located VIIRS pixels in the same overpass
    viirs = [_viirs(0.0, 0.0, 1000, frp=10.0),
             _viirs(0.001, 0.0, 1002, frp=20.0),
             _viirs(0.0, 0.001, 1003, frp=30.0)]
    out = _build(viirs, [_modis(0.0, 0.0, 1000)])
    assert out["n_neigh"].max() == 3
    assert out["frp_sum"].max() == 60.0
    assert out["frp_max"].max() == 30.0


def test_max_scan_filter_drops_swath_edge_pixels():
    viirs = [_viirs(0.0, 0.0, 1000), _viirs(0.001, 0.0, 1000)]
    viirs[1]["scan"] = 0.9
    out = _build(viirs, [_modis(0.0, 0.0, 1000)], max_scan_km=0.5)
    assert len(out) == 1


def test_categorical_confidence_is_ordinal():
    viirs = [_viirs(0.0, 0.0, 1000, conf="l"),
             _viirs(1.0, 1.0, 1000, conf="n"),
             _viirs(2.0, 2.0, 1000, conf="h")]
    out = _build(viirs, [_modis(0.0, 0.0, 1000)])
    assert list(out["conf_ord"]) == [0.0, 1.0, 2.0]


def test_label_is_binary():
    viirs = [_viirs(0.0, 0.0, 1000), _viirs(5.0, 5.0, 1000)]
    out = _build(viirs, [_modis(0.0, 0.0, 1000)])
    assert set(out["modis_detected"].unique()) <= {0, 1}
