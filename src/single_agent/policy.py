import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# TODO: discuss the use of LSTMs
class PpoPolicy(nn.Module):
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

class PpoLstmPolicy(nn.Module):
    '''Recurrent actor-critic for PPO (CleanRL ppo_lstm structure).

    A shared MLP encoder feeds an LSTM whose output drives separate actor and
    critic heads. The recurrence lets the agent INTEGRATE its (S,τ)/(action,Φ)
    measurement history over time to infer direction-to-target — something the
    feedforward PpoPolicy (which only flattens a fixed k-step window) cannot learn.
    The LSTM state is reset at episode boundaries via the `done` mask.
    '''
    def __init__(self, envs, lstm_hidden_size: int = 128):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        # Encoder: 2-layer tanh MLP (matches the project's Andrychowicz-style nets).
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(128, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(lstm_hidden_size, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(lstm_hidden_size, 1), std=1.0)
        self.lstm_hidden_size = lstm_hidden_size

    def initial_state(self, batch_size, device):
        '''Zero (h, c) LSTM state for `batch_size` parallel sequences.'''
        return (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=device),
        )

    def get_states(self, x, lstm_state, done):
        '''Run the encoder+LSTM over a flattened (T·B, obs) batch, resetting the
        recurrent state to zero wherever `done` is set (episode boundaries).
        Returns the per-step hidden features and the final lstm_state.'''
        hidden = self.encoder(x)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))  # (T, B, H)
        done = done.reshape((-1, batch_size))                            # (T, B)
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)          # (T·B, H)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)