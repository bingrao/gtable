# -*- coding: utf-8 -*-
# Copyright 2020 Unknot.id Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gtable.utils.logging import init_logger
from os import makedirs
import random
import numpy as np
import warnings
from os.path import join, exists
import os
from datetime import date
from pathlib import Path
import torch

# BASE_DIR = dirname(dirname(dirname(abspath(__file__))))
BASE_DIR = os.getcwd()


def init_rng(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def create_dir(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


class Context:
    def __init__(self, config):
        # A dictionary of Config Parameters

        self.nums_record = 0
        self.config = config

        # Trainning Device Set up
        self.device = torch.device(self.config.device)
        self.device_id = self.config.cuda_visible_devices
        self.is_cuda = self.config.device == 'cuda'
        self.is_cpu = self.config.device == 'cpu'
        self.is_gpu_parallel = self.is_cuda and (len(self.device_id) > 1)

        if self.is_cuda:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)

        self.project_dir = self.config.project_dir if self.config.project_dir != "" \
            else str(BASE_DIR)

        self.output = self.config.output
        create_dir(self.output)

        self.run_type = self.config.run_type
        self.app = self.config.app
        self.config_file_name = Path(self.config.config).stem if \
            self.config.config is not None else "default"

        if self.config.log_file is not None:
            self.project_log = self.config.log_file
        else:
            self.project_log = join(self.project_dir,
                                    self.output,
                                    'logs',
                                    f'{self.run_type}-{date.today()}-'
                                    f'{self.app.lower()}-'
                                    f'{self.config_file_name}-log.txt')

        if not exists(self.project_log):
            create_dir(os.path.dirname(self.project_log))

        # logger interface
        self.logger = init_logger("log", self.project_log)

        if hasattr(self.config, 'checkpoint_dir'):
            if self.config.checkpoint_dir != "checkpoints":
                self.checkpoint_dir = os.path.join(self.config.checkpoint_dir,
                                                   self.config.app,
                                                   self.config_file_name)
            else:
                self.checkpoint_dir = os.path.join(self.project_dir,
                                                   self.config.checkpoint_dir,
                                                   self.config.app,
                                                   self.config_file_name)
        self.is_master = True

        init_rng(seed=0)
        warnings.filterwarnings('ignore')

        """
        Dataset related helpers
        """

        self.real_data = self.config.real_data

        self.real_name, self.real_ext = os.path.basename(self.real_data).split(".")
        self.fake_data = self.config.fake_data
        self.metadata = self.config.metadata
        self.data_type = self.config.data_type

        self.num_samples = self.config.num_samples
        # self.target_col = self.config.target_col

        self.sep = self.config.sep
        self.drop = [] if self.config.drop is None else self.config.drop
        self.cat_cols = [] if self.config.cat_cols is None else self.config.cat_cols

    def mapping_to_cuda(self, tensor):
        return tensor.to(self.device) if tensor is not None and self.is_cuda else tensor
