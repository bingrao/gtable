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

"""Main library entrypoint."""
from gtable.evaluator import DataEvaluator
from gtable.bin.preprocess import Preprocess
from gtable.data.inputter import pickle_load
from gtable.app import str2app
from os.path import exists
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
        self._model = None
        self._preprocess = None
        self._evaluator = None
        self.real_dataset = None
        self.fake_dataset = None

        # Before runing, we need load the datasets and transformed them if possible
        self.load_dataset()

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

    def load_dataset(self):
        if self.context.real_data is not None and os.path.exists(self.context.real_data):
            self.logging.info(f"Loading real dataset: {self.context.real_data} ...")
            if self._preprocess is None:
                self._preprocess = Preprocess(self.context)
            self.real_dataset = self._preprocess.run(inputPath=self.context.real_data,
                                                     isSave=False)
        else:
            data_path = self.context.save_data + '.train.pkl'
            if os.path.exists(data_path):
                self.logging.info(f"Loading prepared train datasets: {data_path}")
                self.real_dataset = pickle_load(self.context, data_path)
            else:
                self.logging.info(f"The input data does not exist: {data_path}")
                self.real_dataset = None

        if self.context.fake_data is not None and os.path.exists(self.context.fake_data):
            self.logging.info(f"Loading fake dataset: {self.context.fake_data} ...")
            if self._preprocess is None:
                self._preprocess = Preprocess(self.context)
            self.fake_dataset = self._preprocess.run(inputPath=self.context.fake_data,
                                                     isSave=False)

    def evaluate(self):
        """
        Data Preprocess and clean task
        :return: generator dataset for traning task
        """
        self._evaluator.run()

    def generation(self):
        """
        Using trained model to generate anonmymous data
        :return:
        """
        self.model.run(self.real_dataset)

    def run(self):
        if self.run_type == "generation":
            self.generation()
        elif self.run_type == "evaluate":
            self.evaluate()
        else:
            self.logging.info(f"Run type is wrong {self.run_type}")
