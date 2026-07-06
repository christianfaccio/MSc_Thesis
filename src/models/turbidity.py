"""
Beer-Lambert turbidity model.

τ(d) = 1 - exp(-k · |d|),  τ ∈ [0, 1]
"""

import numpy as np

def compute_turbidity(depth, k: float = 0.01):
    """Beer-Lambert depth attenuation. Sign-agnostic in `depth`."""
    return 1.0 - np.exp(-k * np.abs(depth))
