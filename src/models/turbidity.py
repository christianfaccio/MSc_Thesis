"""
Beer-Lambert turbidity model.

τ(d) = 1 - exp(-k · |d|),  τ ∈ [0, 1]

τ = 0 at the surface (clear), → 1 with depth (opaque). The default k = 0.01
[1/m] gives a smooth gradient over a 100 m domain (τ ≈ 0.63 at z = 100 m),
suitable for synthetic-data training. Typical Arabian Gulf / coastal values
are higher (k ≈ 0.1–0.3); switch to those when moving to real CMEMS data.
"""

import numpy as np


def compute_turbidity(depth, k: float = 0.01):
    """Beer-Lambert depth attenuation. Sign-agnostic in `depth`."""
    return 1.0 - np.exp(-k * np.abs(depth))
