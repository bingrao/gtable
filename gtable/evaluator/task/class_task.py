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

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from gtable.evaluator.scores import get_score
from gtable.utils.misc import ClassRegistry
from xgboost import XGBClassifier
from gtable.evaluator.task.task import BaseTask
from gtable.evaluator.task.task import BasedEvaluator
import statistics

_EVALUATE_ClASS_TASK_REGISTRY = ClassRegistry(base_class=BaseTask)
register_class_task = _EVALUATE_ClASS_TASK_REGISTRY.register  # pylint: disable=invalid-name


@register_class_task(name="logistic_regression")
class LogisticRegressionTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(LogisticRegressionTask, self).__init__("logistic_regression", ctx, task_type)

    def build_model(self):
        return LogisticRegression(**self.model_kwargs)


@register_class_task(name="random_forest")
class RandomForestClassifierTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(RandomForestClassifierTask, self).__init__("random_forest", ctx, task_type)

    def build_model(self):
        return RandomForestClassifier(**self.model_kwargs)


@register_class_task(name="decision_tree")
class DecisionTreeClassifierTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(DecisionTreeClassifierTask, self).__init__("decision_tree", ctx, task_type)

    def build_model(self):
        return DecisionTreeClassifier(**self.model_kwargs)


@register_class_task(name="mlp")
class MLPClassifierTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(MLPClassifierTask, self).__init__("mlp", ctx, task_type)

    def build_model(self):
        return MLPClassifier(**self.model_kwargs)


@register_class_task(name="adaboost")
class ADBBoostClassifierTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(ADBBoostClassifierTask, self).__init__("adaboost", ctx, task_type)

    def build_model(self):
        return AdaBoostClassifier(**self.model_kwargs)


@register_class_task(name="xgboost")
class XGBoostTask(BaseTask):
    def __init__(self, ctx, task_type):
        super(XGBoostTask, self).__init__("xgboost", ctx, task_type)

    def build_model(self):
        return XGBClassifier()


def make_evaluate_class_tasks(ctx, task_type):
    names = ctx.config.classify_tasks
    if names is None:
        return []

    if not isinstance(names, list):
        names = [names]
    return [get_evaluate_class_task(name, ctx, task_type) for name in names]


def get_evaluate_class_task(name, ctx, task_type):
    task_class = _EVALUATE_ClASS_TASK_REGISTRY.get(name.lower())
    if task_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return task_class(ctx, task_type)


class ClassEvaluator(BasedEvaluator):
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the
    user to easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods
    of evaluate and the visual evaluation method.
    """

    def __init__(self, ctx, real, fake, seed=1337):
        super(ClassEvaluator, self).__init__(ctx,
                                             real=real,
                                             fake=fake,
                                             seed=seed,
                                             validType="classifer")

    def build_estimators(self):
        return make_evaluate_class_tasks(self.context,
                                         self.metadata['problem_type'])

    def run(self):
        self.logging.info("Build evaluating datasets ...")
        self.build_datasets()

        self.logging.info("Fitting evaluating models ...")
        self.fit_estimators()

        self.logging.info("Getting estimator scores ...")
        estimators_scores = self.score_estimators()

        score_names = [score.name for score in self.scores]
        mape_scores = [get_score("mape").score(estimators_scores[f'{name}_real'],
                                               estimators_scores[f'{name}_fake'])
                       for name in score_names]

        return estimators_scores, 1 - statistics.mean(mape_scores)
