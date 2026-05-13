import gymnasium as gym 
from abc import ABC, abstractmethod 
from SwarmSwIM import Simulator

class BaseEnv(gym.Env):
    def __init__(self):
        # TODO: initialize agent location (random) -> should be abstract right?
        # TODO: observation_space, action_space
        # TODO: all other things
        pass
    
    @abstractmethod 
    def reset(self):
        ...
    
    @abstractmethod 
    def step(self):
        ... 