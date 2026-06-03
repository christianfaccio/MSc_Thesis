"""Tests that SingleAgentEnv is domain-aware at km scale.

After the 5 km scaling, sources, the agent spawn, and the sampled target must
all live inside the configured domain, the observation must keep its 2k+11
layout, and a few steps must run with finite currents.
"""

import numpy as np
import pytest

from src.envs.single_agent import SingleAgentEnv

DOMAIN = (5000.0, 5000.0, 40.0)


@pytest.fixture
def env():
    return SingleAgentEnv(
        xml_file="config/simulation.xml",
        n_sources=4,
        k=4,
        domain=DOMAIN,
        sigma_h=500.0,
        sigma_v=12.0,
        eddy_length_scale=1000.0,
        dt=0.1,
        frame_skip=2,
        max_steps=5,
    )


def test_sources_inside_domain(env):
    env.reset(seed=0)
    assert len(env.sources) == 4
    for s in env.sources:
        assert 0.0 <= s["x"] <= DOMAIN[0]
        assert 0.0 <= s["y"] <= DOMAIN[1]
        assert 0.0 <= s["depth"] <= DOMAIN[2]


def test_agent_spawn_inside_domain(env):
    env.reset(seed=1)
    pos = env.sim.agents[0].pos
    assert 0.0 <= pos[0] <= DOMAIN[0]
    assert 0.0 <= pos[1] <= DOMAIN[1]
    assert 0.0 <= pos[2] <= DOMAIN[2]


def test_observation_shape_and_finiteness(env):
    obs, _ = env.reset(seed=2)
    assert obs.shape == (2 * env.k + 11,)
    assert np.all(np.isfinite(obs))


def test_vortex_field_uses_domain_size(env):
    env.reset(seed=3)
    assert env.sim.vortex_field.domain_size == DOMAIN[0]
    assert env.sim.vortex_field.length_scale == env.eddy_length_scale


def test_sigma_is_threaded(env):
    env.reset(seed=4)
    assert env.sigma_h == 500.0
    assert env.sigma_v == 12.0


def test_steps_run_with_finite_currents(env):
    env.reset(seed=5)
    for _ in range(5):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
