"""

This module implements a DDPG Agent.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

import numpy as np
from torch import nn

from agent import BaseAgent
from config import Config
from tensor import Tensor


class DDPGAgent(BaseAgent):
    def __init__(self, config: Config):
        BaseAgent.__init__(self, config=config, task=config.task)
        self.config = config
        self.task = config.task
        self.network = config.network()
        self.target_network = config.network()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay()
        self.random_process = config.random_process()
        self.total_steps = 0
        self.state = None

    def soft_update(self, target: nn.Module, src: nn.Module):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(
                target_param * (1.0 - self.config.target_network_mix) +
                param * self.config.target_network_mix)

    def evaluation_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return Tensor(action).to_np()

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.environment_info.vector_observations
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [np.random.randn(self.task.action_size)]
        else:
            action = self.network(self.state)
            action = Tensor(action).to_np()
            action += self.random_process.sample()
        action = np.clip(
            action, -1, 1)
        self.task.step(action)
        next_state = self.task.environment_info.vector_observations
        reward = self.task.environment_info.rewards
        done = self.task.environment_info.local_done
        self.record_online_return(reward)
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states = Tensor(transitions.state).torch_tensor_for_device()
            actions = Tensor(transitions.action).torch_tensor_for_device()
            rewards = Tensor(transitions.reward) \
                .torch_tensor_for_device().unsqueeze(-1)
            next_states = Tensor(transitions.next_state) \
                .torch_tensor_for_device()
            mask = Tensor(transitions.mask) \
                .torch_tensor_for_device().unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.act(phi_next)
            q_next = self.target_network.criticize(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.criticize(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_optimization.step()

            phi = self.network.feature(states)
            action = self.network.act(phi)
            policy_loss = -self.network.criticize(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_optimization.step()

            self.soft_update(self.target_network, self.network)
