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
from gtable.utils.evaluate import compute_scores


class BaseSynthesizer:
    """Base class for all default app.
    """
    def __init__(self, ctx):
        self.context = ctx
        self.logging = self.context.logger
        self.config = self.context.config

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(path)
        model.set_device(device)
        return model

    @classmethod
    def from_contex(cls, ctx):
        return cls(ctx)

    def __call__(self, dataset):
        fake_dataset = self.fit_then_sample(dataset)

        self.logging.info("Evaluation %s", self.__class__.__name__)
        scores = compute_scores(dataset, fake_dataset)
        self.logging.info(f"Score: \n {scores}")
