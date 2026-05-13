import numpy as np

# TODO: make reward function more complex and meaningful by adding more components
# like in the Xi et.al. paper (ex. a component related to the pairness with current
# angle to shorten time and save energy)
def reward_func(measured_S: float, measured_tau: float,
                target_S: float, target_tau: float,
                sigma_s: float = 1.0, sigma_tau: float = 1.0) -> float:
    '''
    Computes the reward function for the agent. 
    It is a function of the target (S,tau) couple.

    R = exp( − ((S − S*)/σ_S)²  −  ((τ − τ*)/σ_τ)² )
    '''
    return np.exp(
        -((measured_S - target_S) / sigma_s) ** 2
        - ((measured_tau - target_tau) / sigma_tau) ** 2
    )