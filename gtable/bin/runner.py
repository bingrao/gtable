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

"""Main library entrypoint."""
from gtable.evaluator import DataEvaluator
from gtable.data.dataset import get_data_loader
from gtable.app import str2app
from os.path import exists
import json
import os


def load_app(ctx):
    model_path = ctx.config.checkpoint_path
    ctx.logger.info(f'Loading model from checkpoint {model_path}')
    model = build_base_model(ctx, True, model_path)
    return model


def build_base_model(ctx, checkpoint=False, checkpoint_path=None):
    model = str2app[ctx.config.app].from_contex(ctx)
    # if checkpoint:
    #     checkpoint_mgt = checkpoint_util.Checkpoint.from_config(ctx.config, model)
    #     checkpoint_mgt.restore(checkpoint_path=checkpoint_path, weights_only=True)
    return model


def build_app(ctx):
    model_path = ctx.config.checkpoint_path
    if model_path is not None and exists(model_path):
        ctx.logger.info(f'Loading model from checkpoint {model_path}')
        model = build_base_model(ctx, True, model_path)
    else:
        ctx.logger.info(f'Building model {ctx.config.app} ...')
        model = build_base_model(ctx)
    return model


class Runner(object):
    """Class for running and exporting models."""

    def __init__(self, ctx):
        self.context = ctx
        self.config = self.context.config
        self.logging = self.context.logger
        self.run_type = self.config.run_type

        self.real_dataset = None
        self.fake_dataset = None
        self.metadata = None

        self._model = None
        self._evaluator = None

        # Before runing, we need load the datasets and transformed them if possible
        self.preprocess()

        if self.run_type == "generation":
            self._model = build_app(self.context)

        if self.run_type == "evaluate":
            self._evaluator = DataEvaluator(self.context, self.real_dataset, self.fake_dataset)

    @property
    def model(self):
        return self._model

    @property
    def checkpoint_dir(self):
        """The active model directory."""
        return self.model.checkpoint_dir

    def preprocess(self):
        if self.context.metadata is not None and os.path.exists(self.context.metadata):
            with open(self.context.metadata) as meta_file:
                self.metadata = json.load(meta_file)

        assert self.metadata is not None, \
            f"The input metadata data is empty: {self.context.metadata}"

        if self.context.real_data is not None and os.path.exists(self.context.real_data):
            self.logging.info(f"Loading real dataset: {self.context.real_data} ...")
            self.real_dataset = self.load_dataset(self.context.real_data, self.metadata)
        else:
            self.logging.info(f"The real dataset does not exist: {self.context.real_data}")

        if self.context.fake_data is not None and os.path.exists(self.context.fake_data):
            self.logging.info(f"Loading fake dataset: {self.context.fake_data} ...")
            self.fake_dataset = self.load_dataset(self.context.fake_data, self.metadata)
        else:
            self.logging.info(f"The fake dataset does not exist: {self.context.fake_data}")

        assert self.real_dataset is not None, \
            f"The inpu real data is empty: {self.context.real_data}"

    def load_dataset(self, path, metadata):
        data_loader = get_data_loader(self.context)
        return data_loader(path, metadata)

    def evaluate(self, iteration):
        """
        Data Preprocess and clean task
        :return: generator dataset for traning task
        """
        return self._evaluator.run(iteration)

    def generation(self, iteration):
        """
        Using trained model to generate anonmymous data
        :return:
        """
        return self.model(self.real_dataset, iteration)

    def run(self, iteration=0):
        if self.run_type == "generation":
            return self.generation(iteration)
        elif self.run_type == "evaluate":
            return self.evaluate(iteration)
        else:
            self.logging.info(f"Run type is wrong {self.run_type}")
