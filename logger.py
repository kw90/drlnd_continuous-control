"""

This module implements a Logger.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

import logging
import torch

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


class Logger(object):
    def __init__(self,
                 log_dir: str = './',
                 log_level: int = 0):
        self.log_level = log_level
        self.log_dir = log_dir
        self.vanilla_logger = logging.getLogger()
        self.debug = self.vanilla_logger.debug
        self.info = self.vanilla_logger.info
        self.warning = self.vanilla_logger.warning
        self.error = self.vanilla_logger.error

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v
