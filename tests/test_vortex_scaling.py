"""Tests for the domain-size / eddy-length-scale generalization of VortexField.

The patch must (1) reproduce the legacy 100 m behavior exactly when
domain_size=100, length_scale=1.0; (2) place vortex centers across the whole
domain; (3) stay periodic with period = domain_size; (4) let length_scale
control the eddy radius (where the swirl speed peaks).
"""

import numpy as np
import pytest

from SwarmSwIM import sim_functions


class _StubAgent:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.name = "stub"


def _legacy_current(centers, intensities, px, py):
    """Reference implementation of the ORIGINAL (pre-patch) VortexField math:
    hardcoded 0-100 tiling and the (distance+1)**0.75 metre-scale falloff."""
    x, y = px % 100, py % 100
    current = np.array([0.0, 0.0, 0.0])
    for (xv, yv), intensity in zip(centers, intensities):
        xv, yv = float(xv), float(yv)
        if x - xv > 50: xv += 100
        if x - xv < -50: xv -= 100
        if y - yv > 50: yv += 100
        if y - yv < -50: yv -= 100
        distance = (x - xv) ** 2 + (y - yv) ** 2
        vorticity = intensity / (distance + 1) ** 0.75
        current += np.array([vorticity * (y - yv), -vorticity * (x - xv), 0.0])
    return current


def _make_field(domain_size=100.0, length_scale=1.0, centers=None, intensities=None):
    vf = sim_functions.VortexField(density=3, intensity=0.5,
                                   rng=np.random.default_rng(0),
                                   domain_size=domain_size, length_scale=length_scale)
    if centers is not None:
        vf.vortex_centers = np.asarray(centers, dtype=float)
        vf.random_intensity = np.asarray(intensities, dtype=float)
    return vf


def test_defaults_are_backward_compatible():
    """domain_size=100, length_scale=1.0 must equal the legacy field."""
    vf = sim_functions.VortexField(density=5, intensity=0.5,
                                   rng=np.random.default_rng(42))
    assert vf.domain_size == 100.0
    assert vf.length_scale == 1.0
    for px, py in [(10.0, 20.0), (55.0, 95.0), (150.0, -30.0)]:
        got = vf.current_vortex_calculate(_StubAgent(px, py))
        exp = _legacy_current(vf.vortex_centers, vf.random_intensity, px, py)
        assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)


def test_centers_span_full_domain():
    vf = sim_functions.VortexField(density=2000, intensity=0.3,
                                   rng=np.random.default_rng(1), domain_size=5000.0)
    assert vf.vortex_centers.min() >= 0.0
    assert vf.vortex_centers.max() <= 5000.0
    # With 2000 draws over 5 km, at least one centre should land beyond the
    # legacy 100 m tile — proving centres are no longer pinned to [0, 100].
    assert vf.vortex_centers.max() > 100.0


def test_periodicity_with_domain_size():
    L = 5000.0
    vf = _make_field(domain_size=L, length_scale=1000.0,
                     centers=[[1000.0, 2000.0], [4000.0, 500.0]],
                     intensities=[0.4, -0.3])
    base = vf.current_vortex_calculate(_StubAgent(1234.0, 4321.0))
    shifted = vf.current_vortex_calculate(_StubAgent(1234.0 + L, 4321.0 - L))
    assert shifted == pytest.approx(base, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("length_scale", [50.0, 200.0, 1000.0])
def test_length_scale_sets_eddy_radius(length_scale):
    """A single vortex's swirl speed should peak at a radius that grows with
    length_scale. Use a huge domain so no periodic wrapping interferes."""
    big = 1.0e7
    cx = cy = big / 2
    vf = _make_field(domain_size=big, length_scale=length_scale,
                     centers=[[cx, cy]], intensities=[1.0])
    radii = np.linspace(1.0, 5.0 * length_scale, 400)
    speeds = [np.linalg.norm(vf.current_vortex_calculate(_StubAgent(cx + r, cy)))
              for r in radii]
    r_peak = radii[int(np.argmax(speeds))]
    # Peak of r / (r^2/L^2 + 1)^0.75 is at r = L*sqrt(2) ≈ 1.41 L.
    assert r_peak == pytest.approx(length_scale * np.sqrt(2.0), rel=0.1)
