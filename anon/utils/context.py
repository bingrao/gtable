from anon.utils.logging import init_logger
from os import makedirs
import random
import numpy as np
import warnings
from os.path import dirname, abspath, join, exists
import os

BASE_DIR = dirname(dirname(abspath(__file__)))


def init_rng(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    # torch.random.manual_seed(seed)


def create_dir(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


class Context:
    def __init__(self, config):
        # A dictionary of Config Parameters
        self.config = config

        self.project_dir = self.config.project_dir if self.config.project_dir != "" \
            else str(BASE_DIR)

        self.project_log = self.config.log_file
        if not exists(self.project_log):
            self.project_log = join(os.path.dirname(self.project_dir), 'logs', 'log.txt')
            create_dir(os.path.dirname(self.project_log))

        # logger interface
        self.logger = init_logger("log", self.project_log)

        self.checkpoint_dir = os.path.join(self.config.project_dir, self.config.checkpoint_dir, self.config.app)

        init_rng(seed=0)
        warnings.filterwarnings('ignore')
