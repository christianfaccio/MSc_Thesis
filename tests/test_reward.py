import math

import numpy as np
import pytest

from src.single_agent.reward import reward_func


def test_perfect_hit_returns_one():
    assert reward_func(2.0, 0.5, 2.0, 0.5) == pytest.approx(1.0)


def test_perfect_hit_at_zero_target():
    assert reward_func(0.0, 0.0, 0.0, 0.0) == pytest.approx(1.0)


def test_symmetry_in_salinity_error():
    target_s, target_t = 1.0, 0.3
    above = reward_func(target_s + 0.7, target_t, target_s, target_t)
    below = reward_func(target_s - 0.7, target_t, target_s, target_t)
    assert above == pytest.approx(below)


def test_symmetry_in_turbidity_error():
    target_s, target_t = 1.0, 0.3
    above = reward_func(target_s, target_t + 0.4, target_s, target_t)
    below = reward_func(target_s, target_t - 0.4, target_s, target_t)
    assert above == pytest.approx(below)


def test_bounded_between_zero_and_one():
    target_s, target_t = 2.0, 0.5
    for ds in np.linspace(-5.0, 5.0, 11):
        for dt in np.linspace(-5.0, 5.0, 11):
            r = reward_func(target_s + ds, target_t + dt, target_s, target_t)
            assert 0.0 < r <= 1.0 + 1e-12


def test_known_value_unit_sigma_unit_error():
    r = reward_func(1.0, 0.0, 0.0, 0.0, sigma_s=1.0, sigma_tau=1.0)
    assert r == pytest.approx(math.exp(-1.0))


def test_coupling_independent_axes():
    r = reward_func(1.0, 1.0, 0.0, 0.0, sigma_s=1.0, sigma_tau=1.0)
    assert r == pytest.approx(math.exp(-2.0))


def test_sigma_scaling_salinity():
    # error of 2 with sigma=2 should equal error of 1 with sigma=1
    r_wide = reward_func(2.0, 0.0, 0.0, 0.0, sigma_s=2.0)
    r_unit = reward_func(1.0, 0.0, 0.0, 0.0, sigma_s=1.0)
    assert r_wide == pytest.approx(r_unit)


def test_sigma_scaling_turbidity():
    r_wide = reward_func(0.0, 0.5, 0.0, 0.0, sigma_tau=0.5)
    r_unit = reward_func(0.0, 1.0, 0.0, 0.0, sigma_tau=1.0)
    assert r_wide == pytest.approx(r_unit)


def test_decreases_monotonically_with_error_magnitude():
    target_s, target_t = 0.0, 0.0
    errors = np.linspace(0.0, 5.0, 21)
    rewards = [reward_func(e, 0.0, target_s, target_t) for e in errors]
    assert all(rewards[i] >= rewards[i + 1] for i in range(len(rewards) - 1))
