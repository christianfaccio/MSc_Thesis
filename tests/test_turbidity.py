import math

import numpy as np
import pytest

from src.models.turbidity import compute_turbidity


def test_surface_is_clear():
    assert compute_turbidity(0.0) == pytest.approx(0.0)


def test_sign_agnostic():
    assert compute_turbidity(-50.0) == pytest.approx(compute_turbidity(50.0))


def test_known_value_z_100_k_001():
    val = compute_turbidity(100.0, k=0.01)
    assert val == pytest.approx(1.0 - math.exp(-1.0))


def test_deep_limit_saturates_at_one():
    assert compute_turbidity(1e6) == pytest.approx(1.0)


def test_strictly_monotonic_in_depth():
    depths = np.linspace(0.0, 500.0, 50)
    vals = [compute_turbidity(d) for d in depths]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def test_bounded_between_zero_and_one():
    depths = np.linspace(-1000.0, 1000.0, 101)
    for d in depths:
        v = compute_turbidity(d)
        assert 0.0 <= v < 1.0


def test_custom_k_affects_decay_rate():
    fast = compute_turbidity(50.0, k=0.1)
    slow = compute_turbidity(50.0, k=0.01)
    assert fast > slow


def test_vectorized_input_shape_preserved():
    depths = np.array([0.0, 50.0, 100.0, 200.0])
    vals = compute_turbidity(depths)
    assert vals.shape == depths.shape
    assert vals[0] == pytest.approx(0.0)
