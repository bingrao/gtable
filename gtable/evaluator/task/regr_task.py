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
    def __init__(self, ctx):
        super(RandomForestRegressorTask, self).__init__("random_forest_regr", ctx)

    def build_model(self):
        return RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)


@register_evaluate_regr_task(name="lasso")
class LassoRegressorTask(BaseTask):
    def __init__(self, ctx):
        super(LassoRegressorTask, self).__init__("lasso", ctx)

    def build_model(self):
        return Lasso(random_state=42)


@register_evaluate_regr_task(name="ridge")
class RidgeRegressorTask(BaseTask):
    def __init__(self, ctx):
        super(RidgeRegressorTask, self).__init__("ridge", ctx)

    def build_model(self):
        return Ridge(alpha=1.0, random_state=42)


@register_evaluate_regr_task(name="elastic_net")
class ElasticNetRegressorTask(BaseTask):
    def __init__(self, ctx):
        super(ElasticNetRegressorTask, self).__init__("elastic_net", ctx)

    def build_model(self):
        return ElasticNet(random_state=42)


def make_evaluate_regr_tasks(ctx):
    names = ctx.config.regression_tasks
    if names is None:
        return []

    if not isinstance(names, list):
        names = [names]
    return [get_evaluate_regr_task(name, ctx) for name in names]


def get_evaluate_regr_task(name, ctx):
    task_class = _EVALUATE_REGR_TASK_REGISTRY.get(name.lower())
    if task_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return task_class(ctx)


class RegrEvaluator(BasedEvaluator):
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user to
    easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods
    of evaluate and the visual evaluation method.
    """

    def __init__(self, ctx,
                 real: pd.DataFrame,
                 fake: pd.DataFrame,
                 numerical_columns=None,
                 categorical_columns=None,
                 seed=1337):
        self.context = ctx
        super(RegrEvaluator, self).__init__(config=self.context.config,
                                            logger=self.context.logger,
                                            real=real,
                                            fake=fake,
                                            numerical_columns=numerical_columns,
                                            categorical_columns=categorical_columns,
                                            seed=seed,
                                            validType="regressor")

    def build_estimators(self):
        return make_evaluate_regr_tasks(self.context)

    def run(self) -> float:

        self.build_datasets()
        self.fit_estimators()

        estimators_scores = self.score_estimators()
        self.logging.info(f'Metrics score of Regressor tasks:\n {estimators_scores.to_string()}\n')

        # corr, p = self.scores(self.estimators_scores['real'], self.estimators_scores['fake'])
        # return corr

        return 0.0
