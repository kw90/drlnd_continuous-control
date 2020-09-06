"""

This file runs random actions in the specified environment.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

from task import Task
import numpy as np


if __name__ == '__main__':
    task = Task(environment_name='Reacher', is_training=False)
    task.reset()
    num_agents = len(task.environment_info.agents)
    scores = np.zeros(num_agents)
    states = task.states
    while True:
        actions = np.random.randn(num_agents, task.action_size)
        actions = np.clip(actions, -1, 1)
        task.step(actions)
        scores += task.rewards
        if np.any(task.dones):
            break
    print('Total score (averaged over agents) this episode: {}'.format(
        np.mean(scores)))
    task.close()
