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

from gtable.evaluator.scores import make_scores
from sklearn.model_selection import train_test_split
# from dython.nominal import numerical_encoding
from gtable.evaluator.scores import get_score
import abc
import copy
import pandas as pd
import numpy as np
from gtable.data.inputter import category_to_number


class BaseTask(abc.ABC):
    def __init__(self, name, ctx):
        self._name = name
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, y_true, y_pred):
        return self.model.score(y_true, y_pred)


class BasedEvaluator:
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the
    user to easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different
    methods of evaluate and the visual evaluation method.
    """

    def __init__(self,
                 config,
                 logger,
                 real,
                 fake,
                 numerical_columns=None,
                 categorical_columns=None,
                 seed=1337,
                 validType="classifer"):
        """
        :param real: Real dataset (pd.DataFrame)
        :param fake: Synthetic dataset (pd.DataFrame)
        `viz.plot_correlation_comparison` to indicate your model.
        """
        assert validType == "classifer" or validType == "regressor"
        self.validType = validType

        self.config = config
        self.logging = logger

        self.real = real
        self.fake = fake

        self.target_col = self.config.target_col
        self.features_col = self.config.features_col

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.random_seed = seed

        self.estimators = self.build_estimators()

        self.estimator_names = [type(clf).__name__ for clf in self.estimators]

        for estimator in self.estimators:
            assert hasattr(estimator, 'fit')
            assert hasattr(estimator, 'score')

        self.real_estimators = copy.deepcopy(self.estimators)
        self.fake_estimators = copy.deepcopy(self.estimators)

        self.scores = make_scores(self.config.regression_scores) \
            if self.validType == "regressor" \
            else make_scores(self.config.classify_scores)

    def build_datasets_from_dataframe(self):
        # Convert both datasets to numerical representations and split x and  y
        # real_x = numerical_encoding(self.real.drop([self.target_col], axis=1),
        #                             nominal_columns=self.categorical_columns)
        real_x, _ = category_to_number(self.real.drop([self.target_col], axis=1))

        columns = sorted(real_x.columns.tolist())
        real_x = real_x[columns]

        # fake_x = numerical_encoding(self.fake.drop([self.target_col], axis=1),
        #                             nominal_columns=self.categorical_columns)
        fake_x, _ = category_to_number(self.fake.drop([self.target_col], axis=1))
        for col in columns:
            if col not in fake_x.columns.tolist():
                fake_x[col] = 0
        fake_x = fake_x[columns]

        assert real_x.columns.tolist() == fake_x.columns.tolist(), \
            f'real and fake columns are different: \n{real_x.columns}\n{fake_x.columns}'

        if self.validType == "classifer":
            # Encode real and fake target the same
            real_y, uniques = pd.factorize(self.real[self.target_col])
            mapping = {key: value for value, key in enumerate(uniques)}
            fake_y = np.array([mapping.get(key) for key in self.fake[self.target_col].tolist()])
        else:
            real_y = self.real[self.target_col]
            fake_y = self.fake[self.target_col]

        self.real_x, self.real_y = real_x, real_y
        self.fake_x, self.fake_y = fake_x, fake_y

    def build_datasets(self):
        # For reproducibilty:
        np.random.seed(self.random_seed)

        if isinstance(self.real, pd.DataFrame) and isinstance(self.fake, pd.DataFrame):
            self.build_datasets_from_dataframe()
        else:
            self.real_x = self.real.X
            self.real_y = self.real.y

            self.fake_x = self.fake.X
            self.fake_y = self.fake.y

        self.real_x_train, self.real_x_test, self.real_y_train, self.real_y_test = \
            train_test_split(self.real_x, self.real_y, test_size=0.2)

        self.fake_x_train, self.fake_x_test, self.fake_y_train, self.fake_y_test = \
            train_test_split(self.fake_x, self.fake_y, test_size=0.2)

    def fit_estimators(self):
        """
        Fit self.real_estimators and self.fake_estimators to real and fake data, respectively.
        """
        for i, c in enumerate(self.real_estimators):
            self.logging.info(f'Fitting real {i + 1}: {type(c).__name__}')
            c.fit(self.real_x_train, self.real_y_train)

        for i, c in enumerate(self.fake_estimators):
            self.logging.info(f'Fitting fake: {i + 1}: {type(c).__name__}')
            c.fit(self.fake_x_train, self.fake_y_train)

    def score_estimators(self):
        """
        Get F1 scores of self.real_estimators and self.fake_estimators
        on the fake and real data, respectively.

        :return: dataframe with the results for each estimator on each data test set.
        """
        rows = []
        for real_estimator, fake_estimator, estimator_name in \
                zip(self.real_estimators, self.fake_estimators, self.estimator_names):
            # for dataset, target, dataset_name in zip([self.real_x_test, self.fake_x_test],
            #                                          [self.real_y_test, self.fake_y_test],
            #                                          ['real', 'fake']):
            for dataset, target, dataset_name in zip([self.real_x, self.fake_x],
                                                     [self.real_y, self.fake_y],
                                                     ['real', 'fake']):
                predict_real = real_estimator.predict(dataset)
                predict_fake = fake_estimator.predict(dataset)
                row = {'index': f'{estimator_name}_{dataset_name}'}
                for score in self.scores:
                    row.update(score(target, predict_real, "real"))
                    row.update(score(target, predict_fake, "fake"))

                if self.validType == "classifer":
                    jac_sim = get_score("jaccard_similarity").score(predict_real, predict_fake)
                    row.update({'jaccard_similarity': jac_sim})

                rows.append(row)

        return pd.DataFrame(rows).set_index('index')

    @abc.abstractmethod
    def build_estimators(self):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> float:
        raise NotImplementedError
