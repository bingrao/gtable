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

import torch
from gtable.data.dataset import get_data_loader
from gtable.evaluator import DataEvaluator
from torch.nn import functional
from gtable.utils.evaluate import compute_scores
import numpy as np
import os


class BaseSynthesizer:
    """
    Base class for all default app.
    """
    def __init__(self, ctx):
        self.context = ctx
        self.logging = self.context.logger
        self.config = self.context.config
        self.transformer = None

        self.noise_dim = self.config.noise_dim
        self.batch_size = self.config.batch_size
        self.device = self.context.device

    def fit(self, dataset, categorical_columns=tuple(), ordinal_columns=tuple(), **kwargs):
        raise NotImplementedError

    def sample(self, num_samples, **kwargs):
        raise NotImplementedError

    def fit_then_sample(self, dataset, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.logging.info("Fitting %s", self.__class__.__name__)
        self.fit(dataset, categorical_columns, ordinal_columns)

        self.logging.info("Sampling %s", self.__class__.__name__)
        return self.sample(dataset.num_train_dataset)

    def save(self, path):
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.set_device(cls.device)
        return model

    @classmethod
    def from_contex(cls, ctx):
        return cls(ctx)

    def evaluate(self, real_dataset, fake_dataset, iteration=0, scores=None):
        evaluator = DataEvaluator(self.context, real_dataset, fake_dataset)
        return evaluator.run(iteration, scores)

    def _apply_activate(self, data):
        assert self.transformer is not None
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def __call__(self, dataset, iteration=0):
        self.logging.info(f"Fitting and training {self.__class__.__name__}")

        self.fit(dataset)

        self.logging.info(f"Sampling {self.__class__.__name__}")
        fake_train = self.sample(dataset.num_train_dataset)
        fake_test = self.sample(dataset.num_test_dataset)

        if self.config.output:
            output_path = os.path.join(self.config.output,
                                       f"fake_{self.context.real_name}-"
                                       f"{self.context.app.lower()}-"
                                       f"{self.context.config_file_name}-"
                                       f"{iteration}.{self.context.real_ext}")
            if self.context.real_ext == "npz":
                np.savez(output_path,
                         train=fake_train,
                         test=fake_test,
                         metadata=self.transformer.metadata)
            elif self.context.real_ext == "csv":
                pass
            else:
                pass

        fake_dataset = get_data_loader(self.context, "fake")((fake_train, fake_test),
                                                             dataset.metadata)

        self.logging.info(f"Evaluation {self.__class__.__name__}")

        scores = compute_scores(dataset, fake_dataset)

        scores = self.evaluate(dataset, fake_dataset, iteration, scores)

        scores['iteration'] = iteration
        scores['synthesizer'] = self.context.app
        scores['dataset'] = self.context.real_name

        self.logging.info(f"Evaluation Score: "
                          f"\n {scores[['iteration', 'synthesizer', 'dataset', 'name', 'accuracy', 'f1']]}")

        self.logging.info(f"##################### [{iteration}] Over #####################")

        return scores
