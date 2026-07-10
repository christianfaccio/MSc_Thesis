import numpy as np


def reward_func(measured_S: float, measured_tau: float,
                target_S: float, target_tau: float,
                sigma_s: float = 3.0, sigma_tau: float = 0.3,
                eps_s: float = 0.1, eps_tau: float = 0.01) -> float:
    '''
    Multi-scale proximity potential Φ over the (salinity, turbidity) error space.

        Φ = Σ_scales exp( − ((S − S*)/σ_S)²  −  ((τ − τ*)/σ_τ)² )

    WHY MULTI-SCALE (2026-07-05). A single wide kernel cannot both guide long-range
    navigation AND the precision endgame. The wide kernel (σ_S=3.0, σ_τ=0.3) is
    15–30× the success box (ε_S=0.1, ε_τ=0.01), so around the target it saturates to
    a flat plateau (Φ≈plateau for |ΔS|≲1, |Δτ|≲0.1). Rollout diagnosis of the trained
    IPPO policy showed the agent could null ΔS and Δτ *separately* (66% / 89% of failed
    episodes touched each band) but reached BOTH tolerances simultaneously in 0% of
    them — there was no reward gradient carving the exact joint target, so it drifted
    on the plateau until timeout. The two narrow kernels below (5× and 1.5× the success
    tolerances) restore a gradient all the way into the box without touching the tight
    success test (|ΔS|<ε_S, |Δτ|<ε_τ) — ε_τ is kept tight on purpose: it maps to a
    ~0.3–0.6 m depth band, matching the 0.1 m/step vertical resolution (z_scale=0.1).

    The wide kernel is preserved unchanged, so far-field navigation is unaffected; the
    inner kernels only add pull near the target. Φ ∈ (0, 3]. Potential-based shaping
    invariance (Ng et al. 1999) holds for any state-only Φ, so the optimal policy is
    unchanged — only the learning gradient near the target improves.

    σ_S is sized to the ~10 PSU synthetic span; σ_τ=0.3 gives depth parity (see the
    2026-06-30 tuning note). ε_s / ε_tau default to the env success tolerances.
    '''
    dS = measured_S - target_S
    dT = measured_tau - target_tau

    def _g(ss, st):
        return np.exp(-(dS / ss) ** 2 - (dT / st) ** 2)

    # wide: far-field navigation; mid + sharp: gradient into the tight success box.
    return _g(sigma_s, sigma_tau) + _g(5.0 * eps_s, 5.0 * eps_tau) + _g(1.5 * eps_s, 1.5 * eps_tau)