# -*- coding: utf-8 -*-
# Copyright 2020 Bingbing Rao. All Rights Reserved.
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

from os.path import dirname, abspath, join, exists
import os
import logging

BASE_DIR = dirname(dirname(abspath(__file__)))
logger = logging.getLogger()


def init_logger(run_name="logs", save_log=None, level=logging.INFO):
    log_filename = f'{run_name}.log'
    if save_log is None:
        log_dir = join(BASE_DIR, 'logs')
        if not exists(log_dir):
            os.makedirs(log_dir)
        log_filepath = join(log_dir, log_filename)
    else:
        log_filepath = save_log

    logger = logging.getLogger(run_name)

    logger.setLevel(level)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'w', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


class Logger(object):
    def __init__(self):
        self.run_name="logs"
        self.save_log = None
        self.level = logging.INFO
        self.logging = self.init_logger()

    def init_logger(self):
        log_filename = f'{self.run_name}.log'
        if self.save_log is None:
            log_dir = join(BASE_DIR, 'logs')
            if not exists(log_dir):
                os.makedirs(log_dir)
            log_filepath = join(log_dir, log_filename)
        else:
            log_filepath = self.save_log

        logger = logging.getLogger(self.run_name)

        logger.setLevel(self.level)

        if not logger.handlers:  # execute only if logger doesn't already exist
            file_handler = logging.FileHandler(log_filepath, 'w', 'utf-8')
            stream_handler = logging.StreamHandler(os.sys.stdout)

            formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')

            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        return logger

    def info(self, msg, *args, **kwargs):
        self.logging.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logging.debug(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logging.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logging.error(msg, *args, **kwargs)


