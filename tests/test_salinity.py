import math

import numpy as np
import pytest

from src.models.salinity import compute_salinity_analytical

SIGMA_H = 15.0
SIGMA_V = 10.0


def test_at_source_returns_emission_rate(single_source):
    s = single_source[0]
    val = compute_salinity_analytical(s["x"], s["y"], s["depth"], single_source)
    assert val == pytest.approx(s["Q"])


def test_horizontal_sigma_falloff(single_source):
    s = single_source[0]
    val = compute_salinity_analytical(
        s["x"] + SIGMA_H, s["y"], s["depth"], single_source
    )
    assert val == pytest.approx(s["Q"] * math.exp(-0.5))


def test_vertical_sigma_falloff(single_source):
    s = single_source[0]
    val = compute_salinity_analytical(
        s["x"], s["y"], s["depth"] + SIGMA_V, single_source
    )
    assert val == pytest.approx(s["Q"] * math.exp(-0.5))


def test_combined_sigma_falloff(single_source):
    s = single_source[0]
    val = compute_salinity_analytical(
        s["x"] + SIGMA_H, s["y"], s["depth"] + SIGMA_V, single_source
    )
    assert val == pytest.approx(s["Q"] * math.exp(-1.0))


def test_linear_superposition_of_sources(two_sources):
    a, b = two_sources[0:1], two_sources[1:2]
    point = (30.0, 40.0, 15.0)
    val_a = compute_salinity_analytical(*point, a)
    val_b = compute_salinity_analytical(*point, b)
    val_total = compute_salinity_analytical(*point, two_sources)
    assert val_total == pytest.approx(val_a + val_b)


def test_far_field_decays_to_zero(single_source):
    s = single_source[0]
    val = compute_salinity_analytical(
        s["x"] + 5.0 * SIGMA_H, s["y"], s["depth"], single_source
    )
    assert val < 1e-5 * s["Q"]


def test_non_negative_everywhere(two_sources):
    rng = np.random.default_rng(0)
    points = rng.uniform(-50, 150, size=(50, 3))
    for x, y, z in points:
        assert compute_salinity_analytical(x, y, z, two_sources) >= 0.0


def test_vectorized_input_shape_preserved(single_source):
    xs = np.array([50.0, 60.0, 70.0])
    ys = np.array([50.0, 50.0, 50.0])
    zs = np.array([20.0, 20.0, 20.0])
    vals = compute_salinity_analytical(xs, ys, zs, single_source)
    assert vals.shape == xs.shape
    assert vals[0] == pytest.approx(single_source[0]["Q"])
