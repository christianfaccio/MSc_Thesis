import math

import numpy as np
import pytest

from src.utils.pos_latlon import EARTH_RADIUS, latlon_to_pos, pos_to_latlon

METERS_PER_DEG = math.pi / 180.0 * EARTH_RADIUS  # ≈ 111194.93


def test_origin_invariance():
    lat0, lon0 = 24.5, 54.0
    lat, lon, depth = pos_to_latlon(0.0, 0.0, 0.0, lon0, lat0)
    assert lat == pytest.approx(lat0)
    assert lon == pytest.approx(lon0)
    assert depth == pytest.approx(0.0)


def test_one_degree_north_distance():
    lat0, lon0 = 0.0, 0.0
    x, y, _ = latlon_to_pos(lat0 + 1.0, lon0, 0.0, lon0, lat0)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(METERS_PER_DEG)


def test_one_degree_east_at_equator():
    lat0, lon0 = 0.0, 0.0
    x, y, _ = latlon_to_pos(lat0, lon0 + 1.0, 0.0, lon0, lat0)
    assert x == pytest.approx(METERS_PER_DEG)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_one_degree_east_at_45_north():
    lat0, lon0 = 45.0, 0.0
    x, _, _ = latlon_to_pos(lat0, lon0 + 1.0, 0.0, lon0, lat0)
    assert x == pytest.approx(METERS_PER_DEG * math.cos(math.radians(45.0)))


def test_depth_sign_convention():
    lat0, lon0 = 24.5, 54.0
    _, _, depth = pos_to_latlon(0.0, 0.0, -50.0, lon0, lat0)
    assert depth == pytest.approx(50.0)


def test_round_trip_recovers_original():
    lat0, lon0 = 24.5, 54.0
    rng = np.random.default_rng(42)
    points = rng.uniform(-1000, 1000, size=(20, 3))
    for x, y, z in points:
        # follow the depth convention: z negative downward, depth positive
        z = -abs(z)
        lat, lon, depth = pos_to_latlon(x, y, z, lon0, lat0)
        x2, y2, z2 = latlon_to_pos(lat, lon, depth, lon0, lat0)
        assert x2 == pytest.approx(x, abs=1e-6)
        assert y2 == pytest.approx(y, abs=1e-6)
        assert z2 == pytest.approx(z, abs=1e-6)


def test_latlon_to_pos_inverse_of_pos_to_latlon_at_origin():
    lat0, lon0 = 10.0, 20.0
    x, y, z = latlon_to_pos(lat0, lon0, 0.0, lon0, lat0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)
