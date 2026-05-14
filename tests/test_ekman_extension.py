"""Tests for EkmanSpiral as a 2D-surface → 3D-depth transformer.

The new EkmanSpiral takes a 2D surface current (sum of SwarmSwIM's 5 surface
components, plus an optional wind contribution) and returns the depth-extended
3D current at the agent's depth:

    V(z) = exp(-pi*z/D_E) * R(-sign(f)*pi*z/D_E) * (V_surface + V_wind)
"""

import numpy as np
import pytest

from SwarmSwIM import sim_functions


class _StubAgent:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.name = "stub"


def test_at_surface_returns_input_unchanged():
    ek = sim_functions.EkmanSpiral(wind_speed=0.0, latitude=24.5, eddy_viscosity=0.05)
    surface = np.array([0.3, -0.2, 0.0])
    out = ek.calculate(_StubAgent(z=0.0), surface)
    assert out == pytest.approx(np.array([0.3, -0.2, 0.0]))


@pytest.mark.parametrize("z_frac", [0.25, 0.5, 1.0, 2.0])
def test_decay_magnitude_with_depth(z_frac):
    ek = sim_functions.EkmanSpiral(wind_speed=0.0, latitude=24.5, eddy_viscosity=0.05)
    surface = np.array([1.0, 0.0, 0.0])
    z = z_frac * ek.D_E
    out = ek.calculate(_StubAgent(z=z), surface)
    expected_mag = np.exp(-np.pi * z_frac)
    assert np.hypot(out[0], out[1]) == pytest.approx(expected_mag, rel=1e-9)


def test_rotation_clockwise_in_NH():
    """In the NH, a +x surface current rotates clockwise (toward -y) with depth.
    At z = D_E, rotation angle is -pi, so direction flips to -x."""
    ek = sim_functions.EkmanSpiral(wind_speed=0.0, latitude=24.5, eddy_viscosity=0.05)
    surface = np.array([1.0, 0.0, 0.0])
    out = ek.calculate(_StubAgent(z=ek.D_E), surface)
    decay = np.exp(-np.pi)
    assert out[0] == pytest.approx(-decay, rel=1e-9)
    assert out[1] == pytest.approx(0.0, abs=1e-12)


def test_rotation_counterclockwise_in_SH():
    ek_n = sim_functions.EkmanSpiral(wind_speed=0.0, latitude=24.5, eddy_viscosity=0.05)
    ek_s = sim_functions.EkmanSpiral(wind_speed=0.0, latitude=-24.5, eddy_viscosity=0.05)
    surface = np.array([1.0, 0.0, 0.0])
    # At z = D_E/2, NH rotates by -pi/2 (toward -y); SH rotates by +pi/2 (toward +y).
    z = ek_n.D_E / 2.0
    out_n = ek_n.calculate(_StubAgent(z=z), surface)
    out_s = ek_s.calculate(_StubAgent(z=z), surface)
    assert np.sign(out_n[1]) == -1.0
    assert np.sign(out_s[1]) == +1.0
    assert out_n[1] == pytest.approx(-out_s[1], rel=1e-9)


def test_wind_only_when_surface_is_none():
    """With surface_current=None, output is pure wind-driven Ekman.
    At z=0 the wind contribution sits at +45 deg (NH) from wind_direction."""
    ek = sim_functions.EkmanSpiral(
        wind_speed=5.0, wind_direction=0.0, latitude=24.5, eddy_viscosity=0.05
    )
    out = ek.calculate(_StubAgent(z=0.0), surface_current=None)
    angle = np.arctan2(out[1], out[0])
    assert angle == pytest.approx(np.pi / 4, abs=1e-9)


def test_wind_plus_surface_combines_correctly():
    """Wind contribution must be added to the input surface before transformation.
    At z=0, output = surface + V_wind."""
    ek = sim_functions.EkmanSpiral(
        wind_speed=5.0, wind_direction=0.0, latitude=24.5, eddy_viscosity=0.05
    )
    surface = np.array([0.2, -0.1, 0.0])
    out = ek.calculate(_StubAgent(z=0.0), surface)
    expected = np.array([surface[0] + ek.V_wind[0], surface[1] + ek.V_wind[1], 0.0])
    assert out == pytest.approx(expected)


@pytest.mark.parametrize("lat", [-60.0, -24.5, 24.5, 60.0])
@pytest.mark.parametrize("z", [0.0, 5.0, 50.0])
def test_w_is_always_zero(lat, z):
    ek = sim_functions.EkmanSpiral(wind_speed=3.0, latitude=lat, eddy_viscosity=0.05)
    out = ek.calculate(_StubAgent(z=z), np.array([0.3, 0.4, 0.0]))
    assert out[2] == 0.0


def test_simulator_depth_current_at_decays_with_depth():
    """Integration: the Simulator helper depth_current_at should produce a
    smaller drift at depth than at the surface when ekman is active."""
    from SwarmSwIM import Simulator

    sim = Simulator(timeSubdivision=0.1)
    # Force-enable a known uniform surface current and ekman extension.
    sim.environment["is_uniform_current"] = True
    sim.environment["uniform_current"] = np.array([0.5, 0.0, 0.0])
    sim.environment["is_current_3d"] = True
    sim.environment["current_3d_model"] = "ekman"
    sim.current_3d = sim_functions.EkmanSpiral(
        wind_speed=0.0, latitude=24.5, eddy_viscosity=0.05
    )

    surface_agent = _StubAgent(z=0.0)
    deep_agent = _StubAgent(z=sim.current_3d.D_E / 2.0)
    surf_curr = sim.depth_current_at(surface_agent)
    deep_curr = sim.depth_current_at(deep_agent)

    assert np.hypot(surf_curr[0], surf_curr[1]) == pytest.approx(0.5, rel=1e-9)
    assert np.hypot(deep_curr[0], deep_curr[1]) < np.hypot(surf_curr[0], surf_curr[1])
    assert deep_curr[2] == 0.0
