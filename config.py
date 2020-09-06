"""

This module sets the configuration.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

from torch import device, cuda
from normalizer import RescaleNormalizer


class Config:
    DEVICE = device("cuda:0" if cuda.is_available() else "cpu")

    def __init__(self):
        self.task = None
        self.environment = None
        self.__evaluation_environment = None
        self.evaluation_interval = 0
        self.evaluation_episodes = 0
        self.max_steps = 0

        self.network = None
        self.actor_network = None
        self.critic_network = None
        self.replay = None
        self.discount = None
        self.random_process = None
        self.warm_up = None
        self.target_network_mix = 0.001

        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()

        self.save_interval = 0
        self.log_interval = int(1e3)

    @property
    def evaluation_environment(self):
        return self.__evaluation_environment

    @evaluation_environment.setter
    def evaluation_environment(self, environment):
        self.environment = environment
        self.__evaluation_environment = environment.environment_info
        self.state_size = environment.state_size
        self.action_size = environment.action_size
        self.task_size = environment.name
