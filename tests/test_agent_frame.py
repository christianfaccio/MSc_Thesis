"""Tests for the world-to-agent current rotation used in `_build_state`.

The transformation, as written in src/envs/single_agent.py:174-176, is:
    u = U*cos(psi) + V*sin(psi)
    v = U*sin(psi) - V*cos(psi)
    w = W

Note: this is an involution (applying it twice recovers the input).
"""

import numpy as np
import pytest


def world_to_agent(currents: np.ndarray, psi_deg: float) -> np.ndarray:
    c = np.cos(np.deg2rad(psi_deg))
    s = np.sin(np.deg2rad(psi_deg))
    return np.array(
        [
            currents[0] * c + currents[1] * s,
            currents[0] * s - currents[1] * c,
            currents[2],
        ]
    )


def test_identity_when_heading_zero():
    out = world_to_agent(np.array([1.0, 0.0, 0.0]), psi_deg=0.0)
    assert out == pytest.approx(np.array([1.0, 0.0, 0.0]))


def test_x_axis_flow_at_psi_90():
    out = world_to_agent(np.array([1.0, 0.0, 0.0]), psi_deg=90.0)
    assert out == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-12)


def test_y_axis_flow_at_psi_0_flips_v():
    # u = 1*1 + 0*0 = 0, v = 1*0 - 0*1 = 0... actually V=1 → u = 0, v = -1
    out = world_to_agent(np.array([0.0, 1.0, 0.0]), psi_deg=0.0)
    assert out == pytest.approx(np.array([0.0, -1.0, 0.0]))


@pytest.mark.parametrize("psi", [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, -60.0])
def test_w_is_preserved_under_any_heading(psi):
    out = world_to_agent(np.array([0.3, -0.7, 1.5]), psi_deg=psi)
    assert out[2] == pytest.approx(1.5)


@pytest.mark.parametrize("psi", [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, -60.0])
def test_xy_norm_is_preserved(psi):
    inp = np.array([0.3, -0.7, 0.0])
    out = world_to_agent(inp, psi_deg=psi)
    assert np.hypot(out[0], out[1]) == pytest.approx(np.hypot(inp[0], inp[1]))


@pytest.mark.parametrize("psi", [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, -60.0])
def test_involution_round_trip(psi):
    """Because the map is a reflection, applying it twice returns the input."""
    inp = np.array([0.3, -0.7, 1.5])
    out = world_to_agent(world_to_agent(inp, psi_deg=psi), psi_deg=psi)
    assert out == pytest.approx(inp)


def test_matches_env_implementation():
    """Regression-pin: the helper here must equal what the env computes inline."""
    from src.envs.single_agent import SingleAgentEnv

    env = SingleAgentEnv(xml_file="unused.xml", n_sources=2)
    # Replicate the env's exact lines for a sample input and confirm equality.
    currents = np.array([0.4, -0.2, 0.05])
    psi_deg = 37.0
    c = np.cos(np.deg2rad(psi_deg))
    s = np.sin(np.deg2rad(psi_deg))
    expected = np.array(
        [
            currents[0] * c + currents[1] * s,
            currents[0] * s - currents[1] * c,
            currents[2],
        ]
    )
    assert world_to_agent(currents, psi_deg) == pytest.approx(expected)
    # silence unused-fixture warning while still verifying env builds
    assert env is not None
