"""

This module implements a Logger.

Author: Kai Waelti

Inspired by ShangtongZhang's modular implementation of various Deep
Reinforcement Learning Algorithms found at
https://github.com/ShangtongZhang/DeepRL

"""

import logging
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


class Logger(object):
    def __init__(self,
                 log_dir: str = './',
                 log_level: int = 0):
        self.log_level = log_level
        self.log_dir = log_dir
        self.writer = None
        self.vanilla_logger = logging.getLogger()
        self.debug = self.vanilla_logger.debug
        self.info = self.vanilla_logger.info
        self.warning = self.vanilla_logger.warning
        self.error = self.vanilla_logger.error


    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v


    def add_scalar(self, tag, value, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)


    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)