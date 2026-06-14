import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class IppoPolicy(nn.Module):
    '''
    Parameter-shared actor-critic for IPPO.

    Both the actor and the critic take the agent's LOCAL observation: IPPO is
    fully decentralized (de Witt et al. 2020), so there is no centralized critic.
    A single instance is shared across all agents (parameter sharing), which is
    the only "multi-agent" ingredient on top of single-agent PPO.

    Inputs are flat tensors of shape (batch, local_dim); the batch axis collapses
    (num_steps, num_envs, n_agents) — every agent-step is an independent sample.
    '''
    def __init__(self, local_dim: int, n_actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(local_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(local_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class MappoPolicy(nn.Module):
    '''
    Parameter-shared actor-critic for MAPPO (CTDE, Yu et al. 2022).

    The actor takes the agent's LOCAL observation (decentralized execution); the
    critic takes the GLOBAL state (centralized training). Switching IPPO -> MAPPO
    is exactly this: feed the critic the global state instead of the local obs.
    '''
    def __init__(self, local_dim: int, global_dim: int, n_actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(global_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(local_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        )

    def get_value(self, global_state):
        return self.critic(global_state)

    def get_action_and_value(self, x, global_state, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(global_state)
