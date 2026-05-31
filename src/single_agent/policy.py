import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# TODO: discuss the use of LSTMs
class CustomPolicy(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # NOTE: suggestions from Andrychowicz et. al paper:
        # - try a wider value network wrt the policy one (e.g. 256 vs 128/64)
        # - separate networks
        # - 2 hidden layers
        # - tanh activation
        # 
        # TODO: It is HIGHLY suggested (even though they use a continuous action space)
        # to initialize the networks such that the action distribution is centered 
        # in 0. This can be achieved by setting smaller values to the last layer, with 0.5 
        # as best overall std value. The authors suggest to set the last layer weights 
        # with 100x smaller values. 
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
