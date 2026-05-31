import itertools

import numpy as np
import pytest

from src.envs.single_agent import SingleAgentEnv


@pytest.fixture
def env():
    return SingleAgentEnv(xml_file="unused.xml", n_sources=2)


def test_action_table_shape(env):
    table = np.asarray(env._action_to_direction)
    assert table.shape == (27, 3)


def test_zero_action_is_no_op(env):
    table = list(itertools.product([-1, 0, 1], repeat=3))
    zero_idx = table.index((0, 0, 0))
    vec = np.asarray(env._action_to_direction[zero_idx])
    assert np.allclose(vec, [0.0, 0.0, 0.0])


def test_all_nonzero_actions_are_unit_norm(env):
    table_raw = list(itertools.product([-1, 0, 1], repeat=3))
    zero_idx = table_raw.index((0, 0, 0))
    for i, vec in enumerate(env._action_to_direction):
        if i == zero_idx:
            continue
        assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_specific_corner_diagonal(env):
    table_raw = list(itertools.product([-1, 0, 1], repeat=3))
    idx = table_raw.index((-1, -1, -1))
    vec = np.asarray(env._action_to_direction[idx])
    expected = np.full(3, -1.0 / np.sqrt(3))
    assert np.allclose(vec, expected)


def test_specific_axis_unit_vector(env):
    table_raw = list(itertools.product([-1, 0, 1], repeat=3))
    idx = table_raw.index((1, 0, 0))
    vec = np.asarray(env._action_to_direction[idx])
    assert np.allclose(vec, [1.0, 0.0, 0.0])


def test_action_table_is_deterministic():
    env_a = SingleAgentEnv(xml_file="unused.xml", n_sources=2)
    env_b = SingleAgentEnv(xml_file="unused.xml", n_sources=2)
    assert np.allclose(env_a._action_to_direction, env_b._action_to_direction)


def test_all_norms_in_zero_or_one(env):
    for vec in env._action_to_direction:
        n = np.linalg.norm(vec)
        assert n == pytest.approx(0.0) or n == pytest.approx(1.0)


def test_observation_space_dim_matches_formula():
    for k in (1, 4, 8):
        env = SingleAgentEnv(xml_file="unused.xml", n_sources=2, k=k)
        assert env.observation_space.shape == (2 * k + 7,)
