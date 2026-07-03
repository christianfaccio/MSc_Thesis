import numpy as np

def reward_func(measured_S: float, measured_tau: float,
                target_S: float, target_tau: float,
                sigma_s: float = 1.5, sigma_tau: float = 0.3) -> float:
    '''
    Computes the reward function for the agent.

    R = exp( − ((S − S*)/σ_S)²  −  ((τ − τ*)/σ_τ)² )
    '''
    return np.exp(
        -((measured_S - target_S) / sigma_s) ** 2
        - ((measured_tau - target_tau) / sigma_tau) ** 2
    )