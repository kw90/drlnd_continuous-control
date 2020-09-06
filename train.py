"""

This file trains a specified agent in the environment.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

import time
from task import Task
from config import Config
from models import ActorCriticArchitecture
from replay import PrioritizedReplay
from random_process import OrnsteinUhlenbeckProcess
from schedule import LinearSchedule
from agent import BaseAgent
from ddpg_agent import DDPGAgent


def create_ddpg_continuous_config() -> Config:
    config = Config()
    config.task = Task(environment_name='Reacher', is_training=True)
    config.evaluation_interval = int(1e4)
    config.evaluation_episodes = 20
    config.max_steps = int(1e6)
    config.network = lambda: ActorCriticArchitecture(
        config.state_size,
        config.action_size)
    config.replay = lambda: PrioritizedReplay(
        memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_size, ), std=LinearSchedule(start=0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    config.task.reset()
    config.evaluation_environment = config.task
    return config


def run_steps_with(agent: BaseAgent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and \
                not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (
                agent_name, config.tag, agent.total_steps))
        if config.log_interval and \
                not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (
                agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.evaluation_interval and \
                not agent.total_steps % config.evaluation_interval:
            agent.evaluation_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


if __name__ == '__main__':
    config = create_ddpg_continuous_config()
    run_steps_with(DDPGAgent(config))
