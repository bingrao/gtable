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

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, ctx):
        super(LogisticRegressionTask, self).__init__("logistic_regression", ctx)

    def build_model(self):
        return LogisticRegression(multi_class='auto',
                                  solver='lbfgs',
                                  max_iter=500,
                                  random_state=42)


@register_class_task(name="random_forest")
class RandomForestClassifierTask(BaseTask):
    def __init__(self, ctx):
        super(RandomForestClassifierTask, self).__init__("random_forest", ctx)

    def build_model(self):
        return RandomForestClassifier(n_estimators=10, random_state=42)


@register_class_task(name="decision_tree")
class DecisionTreeClassifierTask(BaseTask):
    def __init__(self, ctx):
        super(DecisionTreeClassifierTask, self).__init__("decision_tree", ctx)

    def build_model(self):
        return DecisionTreeClassifier(random_state=42)


@register_class_task(name="mlp")
class MLPClassifierTask(BaseTask):
    def __init__(self, ctx):
        super(MLPClassifierTask, self).__init__("mlp", ctx)

    def build_model(self):
        return MLPClassifier([50, 50], solver='adam',
                             activation='relu',
                             learning_rate='adaptive',
                             random_state=42)


@register_class_task(name="xgboost")
class XGBoostTask(BaseTask):
    def __init__(self, ctx):
        self.context = ctx
        super(XGBoostTask, self).__init__("xgboost", ctx)

    def build_model(self):
        return XGBClassifier()


def make_evaluate_class_tasks(ctx):
    names = ctx.config.classify_tasks

    if names is None:
        return []

    if not isinstance(names, list):
        names = [names]
    return [get_evaluate_class_task(name, ctx) for name in names]


def get_evaluate_class_task(name, ctx):
    task_class = _EVALUATE_ClASS_TASK_REGISTRY.get(name.lower())
    if task_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return task_class(ctx)


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
        return make_evaluate_class_tasks(self.context)

    def run(self) -> float:

        self.logging.info("Build evaluating datasets ...")
        self.build_datasets()

        self.logging.info("Fitting evaluating models ...")
        self.fit_estimators()

        self.logging.info("Getting estimator scores ...")
        estimators_scores = self.score_estimators()

        self.logging.info(f'Metrics score of Classifier tasks:\n{estimators_scores.to_string()}\n')

        score_names = [score.name for score in self.scores]
        mape_scores = [get_score("mape").score(estimators_scores[f'{name}_real'],
                                               estimators_scores[f'{name}_fake'])
                       for name in score_names]

        return 1 - statistics.mean(mape_scores)
