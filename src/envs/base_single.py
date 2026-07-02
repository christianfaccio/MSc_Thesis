from .base import BaseEnv


class BaseSingleAgentEnv(BaseEnv):
    '''Concrete single-agent env on the synthetic Gaussian-field baseline.

    BaseEnv provides reset/step as abstract-with-body; this thin subclass makes it
    instantiable and accepts (and ignores) the gym `options` kwarg so the standard
    Gymnasium wrappers and SyncVectorEnv work unchanged.
    '''

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed)

    def step(self, action):
        return super().step(action)
