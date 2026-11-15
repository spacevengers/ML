import numpy as np

from terrawatch.geo import (block_id, chord_for_km, haversine_km,
                            SpatialTimeIndex, to_xyz)


def test_xyz_is_unit_length():
    xyz = to_xyz([0, 45, -30], [0, 90, 120])
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0)


def test_haversine_matches_known_distance():
    # Nairobi -> Mombasa is roughly 440 km
    d = haversine_km(-1.2921, 36.8219, -4.0435, 39.6682)
    assert 400 < float(d) < 500


def test_haversine_zero_for_same_point():
    assert haversine_km(0.0, 0.0, 0.0, 0.0) == 0.0


def test_chord_is_monotonic_in_radius():
    assert chord_for_km(1.0) < chord_for_km(2.0) < chord_for_km(10.0)


def test_block_id_groups_nearby_points():
    a = block_id([-1.2], [36.8], deg=1.0)[0]
    b = block_id([-1.9], [36.5], deg=1.0)[0]   # same 1x1 degree cell
    c = block_id([1.5], [36.8], deg=1.0)[0]
    assert a == b
    assert a != c


def test_spatial_time_index_respects_both_constraints():
    idx = SpatialTimeIndex(lat=[0.0], lon=[0.0], minutes=[1000])

    # ~111 m away, 10 min later -> match
    hits = list(idx.query([0.001], [0.0], [1010], radius_km=1.5, window_min=30))
    assert hits[0].size == 1

    # same place but 10 hours later -> no match
    hits = list(idx.query([0.001], [0.0], [1600], radius_km=1.5, window_min=30))
    assert hits[0].size == 0

    # ~22 km away -> no match
    hits = list(idx.query([0.2], [0.0], [1010], radius_km=1.5, window_min=30))
    assert hits[0].size == 0


def test_any_within_returns_binary_mask():
    idx = SpatialTimeIndex(lat=[0.0, 5.0], lon=[0.0, 5.0], minutes=[1000, 1000])
    mask = idx.any_within([0.0, 9.0], [0.0, 9.0], [1000, 1000], 1.5, 30)
    assert list(mask) == [1, 0]
