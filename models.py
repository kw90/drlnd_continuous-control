"""

This module implements a Deterministic Actor Critic Network using PyTorch.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable
from config import Config
from tensor import Tensor


class LayerFill():
    def __init__(self, layer: nn.Linear):
        self.layer = layer

    def fill_input_tensor_scale(self, w_scale: float = 1.0):
        nn.init.orthogonal_(self.layer.weight.data)
        self.layer.weight.data.mul_(w_scale)
        nn.init.constant_(self.layer.bias.data, 0)
        return self.layer


class DummyNetwork(nn.Module):
    def __init__(self, state_size: int):
        super(DummyNetwork, self).__init__()
        self.feature_dimensions = state_size

    def forward(self, x):
        return x


class FullyConnectedNetwork(nn.Module):
    """Generic Class for a Fully Connected Network for an Actor and Critic."""

    def __init__(self,
                 state_size: int,
                 seed: int = 42,
                 hidden_units: Tuple[int] = (64, 64),
                 gate: Callable[[torch.Tensor, bool], torch.Tensor] = F.relu,
                 layer_init: Callable = None):
        """
        Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            hidden_units (Tuple[int]): # Nodes in hidden layers
            gate (Callable): Non-linear activation function
            layer_init (Callable): Function(s) to fill input Tensor(s)
                (e.g. layer.weight and layer.bias)
        """
        super(FullyConnectedNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        dimensions = (state_size, ) + hidden_units
        self.layers = [nn.Linear(layer_in, layer_out)
                       for layer_in, layer_out
                       in zip(dimensions[:-1], dimensions[1:])]
        if layer_init is None:
            [LayerFill(layer=layer).fill_input_tensor_scale(1.0)
             for layer
             in self.layers]
        else:
            [layer_init(layer) for layer in self.layers]
        self.feature_dimensions = dimensions[-1]

    def forward(self, x):
        """Build a network that maps state -> action values."""
        return [self.gate(layer(x)) for layer in self.layers]


class ActorCriticArchitecture(nn.Module):
    """Module that maps states to action values."""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int = 42,
                 phi_network: nn.Module = None,
                 actor_network: nn.Module = None,
                 critic_network: nn.Module = None):
        """
        Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of possible actions
            seed (int): Random seed
        """
        super(ActorCriticArchitecture, self).__init__()
        self.seed = torch.manual_seed(seed)
        if phi_network is None:
            phi_network = DummyNetwork(state_size=state_size)
        if actor_network is None:
            actor_network = FullyConnectedNetwork(
                state_size=state_size,
                seed=seed,
                hidden_units=(400, 300))
        if critic_network is None:
            critic_network = FullyConnectedNetwork(
                state_size=state_size + action_size,
                seed=seed,
                hidden_units=(400, 300))
        self.phi_network = phi_network
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.fully_connected_action = LayerFill(
            nn.Linear(
                self.actor_network.feature_dimensions,
                action_size)
                ).fill_input_tensor_scale(1e-3)
        self.fully_connected_critic = LayerFill(
            nn.Linear(
                self.critic_network.feature_dimensions,
                1)
                ).fill_input_tensor_scale(1e-3)
        self.phi_params = list(self.phi_network.parameters())
        self.actor_params = list(self.actor_network.parameters()) + \
            list(self.fully_connected_action.parameters())
        self.critic_params = list(self.critic_network.parameters()) + \
            list(self.fully_connected_critic.parameters())
        self.actor_optimization = torch.optim.Adam(
            self.actor_params + self.phi_params, lr=1e-3)
        self.critic_optimization = torch.optim.Adam(
            self.critic_params + self.phi_params, lr=1e-3)
        self.to(Config.DEVICE)

    def feature(self, observation):
        observation = Tensor(observation).torch_tensor_for_device()
        return self.phi_network(observation)

    def act(self, phi):
        x = torch.stack(self.actor_network(phi)).to(Config.DEVICE)
        return torch.tanh(self.fully_connected_action(x))

    def criticize(self, phi, action):
        return self.fully_connected_critic(self.critic_network(
            torch.cat([phi, action], dim=1)))

    def forward(self, observation):
        phi = self.feature(observation=observation)
        return self.act(phi)
