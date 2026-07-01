import numpy as np

def reward_func(measured_S: float, measured_tau: float,
                target_S: float, target_tau: float,
                sigma_s: float = 1.5, sigma_tau: float = 0.3) -> float:
    '''
    Computes the reward function for the agent.

    R = exp( − ((S − S*)/σ_S)²  −  ((τ − τ*)/σ_τ)² )

    The sigmas are used to balance the two components,
    since the salinity scale is more or less 10x the
    turbidity one, while the exponential is used to
    have a final value between 0 and 1.

    σ_S is sized to the Oceananigans salinity span (~1.3 PSU): widened from 0.5 to
    1.5 (2026-06-28) so the shaping potential Φ varies across the WHOLE domain rather
    than collapsing to ~0 away from a rare tail target (with σ_S=0.5, Φ at the spawn
    ΔS≈1.3 was exp(-(1.3/0.5)²)≈0.001 — effectively sparse; at σ_S=1.5 it is ≈0.47).
    This only changes the dense guidance, not the success test (|ΔS|<ε_S, separate).

    σ_τ tightened 0.8 → 0.3 (2026-06-30): turbidity only spans ~0.33 over the 40 m
    column (τ=1−exp(−0.01|z|)), so σ_τ=0.8 left the depth dimension of Φ nearly flat
    (worst-depth factor 0.84 vs salinity's 0.47 — 3.4× less pull), and the agent
    learned to ignore depth (z ping-pong). σ_τ=0.3 gives parity with σ_S (≈range/1.1):
    worst-depth factor drops to ≈0.30, restoring a real depth-seeking gradient.

    TODO: to discuss using a linear function instead
    of an exponential.
    '''
    return np.exp(
        -((measured_S - target_S) / sigma_s) ** 2
        - ((measured_tau - target_tau) / sigma_tau) ** 2
    )