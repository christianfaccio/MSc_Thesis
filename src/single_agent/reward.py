import numpy as np

# 
def reward_func(measured_S: float, measured_tau: float,
                target_S: float, target_tau: float,
                sigma_s: float = 2.0, sigma_tau: float = 0.8) -> float:
    '''
    Computes the reward function for the agent.

    R = exp( − ((S − S*)/σ_S)²  −  ((τ − τ*)/σ_τ)² )

    The sigmas are used to balance the two components,
    since the salinity scale is more or less 10x the 
    turbidity one, while the exponential is used to
    have a final value between 0 and 1.

    TODO: to discuss using a linear function instead 
    of an exponential.
    '''
    return np.exp(
        -((measured_S - target_S) / sigma_s) ** 2
        - ((measured_tau - target_tau) / sigma_tau) ** 2
    )