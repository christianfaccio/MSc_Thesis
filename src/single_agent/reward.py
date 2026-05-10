from models.turbidity import turbidity_model
import numpy as np

def reward_func(S, tau, w1, w2):
    '''
    Computes the reward function for the agent. 

    R = w1 * f(S) + w2 * f(tau)   // normalized
    '''
    return w1 * S + w2 * tau # TODO: modify this function