"""

This module implements a Task for a Unity ML Agent Environment.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

import numpy as np
from unityagents import UnityEnvironment


class Task():
    def __init__(self,
                 environment_name: str = 'Reacher',
                 is_training: bool = True,
                 num_envs: int = 1,
                 is_single_process: bool = True):
        self.name = environment_name
        self.environment = UnityEnvironment(environment_name)
        # get the default brain
        self.brain_name = self.environment.brain_names[0]
        self.brain = self.environment.brains[self.brain_name]
        self.is_training = is_training

    def reset(self):
        if self.is_training:
            self.environment_info = self.environment.reset(
                train_mode=True)[self.brain_name]
        else:
            self.environment_info = self.environment.reset(
                train_mode=False)[self.brain_name]
        self.states = self.environment_info.vector_observations
        self.state_size = int(self.states.shape[1])
        self.action_size = self.brain.vector_action_space_size
        self.rewards = self.environment_info.rewards
        self.dones = self.environment_info.local_done

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        self.environment_info = self.environment.step(
            actions)[self.brain_name]

    def close(self):
        self.environment.close()
