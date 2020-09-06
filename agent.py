"""

This module implements a Base Agent for DDPG, D4PG, A3C, PPO, ...

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

import torch
import pickle
import numpy as np
from config import Config
from task import Task
from logger import Logger


class BaseAgent():
    def __init__(self, config: Config, task: Task):
        self.config = config
        self.task = task
        self.logger = Logger()

    def save(self, filename: str):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename: str):
        state_dict = torch.load(
            '%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def close(self):
        self.task.close()

    def evaluation_step(self, state):
        raise NotImplementedError

    def evaluation_episode(self):
        task = self.config.task
        task.reset()
        while True:
            action = self.evaluation_step(task.states)
            state, reward, done, info = task.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def evaluation_episodes(self):
        episodic_returns = []
        for ep in range(self.config.evaluation_episodes):
            total_rewards = self.evaluation_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps,
            np.mean(episodic_returns),
            np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar(
            'episodic_return_test',
            np.mean(episodic_returns),
            self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }
