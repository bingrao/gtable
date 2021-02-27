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

from gtable.utils.misc import ClassRegistry
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from gtable.evaluator.task.task import BaseTask
from gtable.evaluator.task.task import BasedEvaluator
import pandas as pd


_EVALUATE_REGR_TASK_REGISTRY = ClassRegistry(base_class=BaseTask)
register_evaluate_regr_task = _EVALUATE_REGR_TASK_REGISTRY.register  # pylint: disable=invalid-name


@register_evaluate_regr_task(name="random_forest_regr")
class RandomForestRegressorTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(RandomForestRegressorTask, self).__init__("random_forest_regr", ctx, task_type)

    def build_model(self):
        return RandomForestRegressor(**self.model_kwargs)


@register_evaluate_regr_task(name="lasso")
class LassoRegressorTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(LassoRegressorTask, self).__init__("lasso", ctx, task_type)

    def build_model(self):
        return Lasso(**self.model_kwargs)


@register_evaluate_regr_task(name="ridge")
class RidgeRegressorTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(RidgeRegressorTask, self).__init__("ridge", ctx, task_type)

    def build_model(self):
        return Ridge(**self.model_kwargs)


@register_evaluate_regr_task(name="elastic_net")
class ElasticNetRegressorTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(ElasticNetRegressorTask, self).__init__("elastic_net", ctx, task_type)

    def build_model(self):
        return ElasticNet(**self.model_kwargs)


def make_evaluate_regr_tasks(ctx, task_type):
    names = ctx.config.regression_tasks
    if names is None:
        return []

    if not isinstance(names, list):
        names = [names]
    return [get_evaluate_regr_task(name, ctx, task_type) for name in names]


def get_evaluate_regr_task(name, ctx, task_type):
    task_class = _EVALUATE_REGR_TASK_REGISTRY.get(name.lower())
    if task_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return task_class(ctx, task_type)


class RegrEvaluator(BasedEvaluator):
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user to
    easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods
    of evaluate and the visual evaluation method.
    """

    def __init__(self, ctx, real, fake, seed=1337):
        super(RegrEvaluator, self).__init__(ctx,
                                            real=real,
                                            fake=fake,
                                            seed=seed,
                                            validType="regressor")

    def build_estimators(self):
        return make_evaluate_regr_tasks(self.context, self.metadata['problem_type'])

    def run(self):
        self.build_datasets()
        self.fit_estimators()

        estimators_scores = self.score_estimators()

        # corr, p = self.scores(self.estimators_scores['real'], self.estimators_scores['fake'])
        # return corr

        return estimators_scores, 0.0
