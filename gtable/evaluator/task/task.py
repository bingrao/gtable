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
from gtable.evaluator.scores import get_score
import abc
import copy
import pandas as pd
import numpy as np
from gtable.data.inputter import category_to_number
from gtable.utils.constants import NUMERICAL, CATEGORICAL, ORDINAL


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
                 ctx,
                 real,
                 fake,
                 seed=1337,
                 validType="classifer"):
        """
        :param real: Real dataset
        :param fake: Synthetic dataset
        `viz.plot_correlation_comparison` to indicate your model.
        """
        assert validType == "classifer" or validType == "regressor"

        self.context = ctx
        self.config = self.context.config
        self.logging = self.context.logger
        self.validType = validType

        self.real = real
        self.fake = fake

        self.metadata = real.metadata

        # self.target_col = self.real.metadata['label']
        self.features_col = self.config.features_col

        self.numerical_cols = [item['name'] for item in real.metadata['columns']
                               if item['type'] == NUMERICAL]

        self.categorial_cols = [item['name'] for item in real.metadata['columns']
                                if item['type'] == CATEGORICAL]

        self.ordinal_cols = [item['name'] for item in real.metadata['columns']
                             if item['type'] == ORDINAL]

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

    def build_datasets_from_dataframe(self, real, fake):
        # Convert both datasets to numerical representations and split x and  y
        # real_x = numerical_encoding(self.real.drop([self.target_col], axis=1),
        #                             nominal_columns=self.categorical_columns)
        real_x, _ = category_to_number(real.drop([self.target_col], axis=1))

        columns = sorted(real_x.columns.tolist())
        real_x = real_x[columns]

        # fake_x = numerical_encoding(self.fake.drop([self.target_col], axis=1),
        #                             nominal_columns=self.categorical_columns)
        fake_x, _ = category_to_number(fake.drop([self.target_col], axis=1))
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

        self.real_x_train, self.real_x_test, self.real_y_train, self.real_y_test = \
            train_test_split(real_x, real_y, test_size=0.2)

        self.fake_x_train, self.fake_x_test, self.fake_y_train, self.fake_y_test = \
            train_test_split(fake_x, fake_y, test_size=0.2)

    def build_datasets_from_dataset(self, real, fake):
        self.real_x_train, self.real_y_train, \
        self.real_x_test, self.real_y_test = real.split_dataset()

        self.fake_x_train, self.fake_y_train, \
        self.fake_x_test, self.fake_y_test = fake.split_dataset()

    def build_datasets(self):
        # For reproducibilty:
        np.random.seed(self.random_seed)

        if isinstance(self.real, pd.DataFrame) and isinstance(self.fake, pd.DataFrame):
            self.build_datasets_from_dataframe(self.real, self.fake)
        else:
            self.build_datasets_from_dataset(self.real, self.fake)

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

    # def score_estimators(self):
    #     """
    #     Get F1 scores of self.real_estimators and self.fake_estimators
    #     on the fake and real data, respectively.
    #
    #     :return: dataframe with the results for each estimator on each data test set.
    #     """
    #     rows = []
    #     for real_estimator, fake_estimator, estimator_name in \
    #             zip(self.real_estimators, self.fake_estimators, self.estimator_names):
    #         for dataset, target, dataset_name in zip([self.real_x_test, self.fake_x_test],
    #                                                  [self.real_y_test, self.fake_y_test],
    #                                                  ['real', 'fake']):
    #             predict_real = real_estimator.predict(dataset)
    #             predict_fake = fake_estimator.predict(dataset)
    #             row = {'name': f'{estimator_name}_{dataset_name}'}
    #             for score in self.scores:
    #                 row.update(score(target, predict_real, "real"))
    #                 row.update(score(target, predict_fake, "fake"))
    #
    #             if self.validType == "classifer":
    #                 jac_sim = get_score("jaccard_similarity").score(predict_real, predict_fake)
    #                 row.update({'jaccard_similarity': jac_sim})
    #
    #             rows.append(row)
    #     output = pd.DataFrame(rows)
    #     return output

    # def score_estimators(self):
    #     """
    #     Get F1 scores of self.real_estimators and self.fake_estimators
    #     on the fake and real data, respectively.
    #
    #     :return: dataframe with the results for each estimator on each data test set.
    #     """
    #     rows = []
    #     for real_estimator, estimator_name in zip(self.real_estimators, self.estimator_names):
    #         row = {'name': f'{estimator_name}_real'}
    #         for dataset, target, dataset_name in zip([self.real_x_test, self.fake_x_test],
    #                                                  [self.real_y_test, self.fake_y_test],
    #                                                  ['real', 'fake']):
    #             predict_real = real_estimator.predict(dataset)
    #             for score in self.scores:
    #                 row.update(score(target, predict_real, dataset_name))
    #         rows.append(row)
    #
    #     for fake_estimator, estimator_name in zip(self.fake_estimators, self.estimator_names):
    #         row = {'name': f'{estimator_name}_fake'}
    #         for dataset, target, dataset_name in zip([self.real_x_test, self.fake_x_test],
    #                                                  [self.real_y_test, self.fake_y_test],
    #                                                  ['real', 'fake']):
    #             predict_fake = fake_estimator.predict(dataset)
    #             for score in self.scores:
    #                 row.update(score(target, predict_fake, dataset_name))
    #
    #         rows.append(row)
    #
    #     output = pd.DataFrame(rows)
    #     return output

    def score_estimators(self):
        """
        Get F1 scores of self.real_estimators and self.fake_estimators
        on the fake and real data, respectively.

        :return: dataframe with the results for each estimator on each data test set.
        """
        rows = []
        predict_real = {}
        predict_fake = {}
        for real_estimator, fake_estimator, estimator_name in \
            zip(self.real_estimators, self.fake_estimators, self.estimator_names):
            # Using model trained by real dataset to predict real and fake test datasets
            row = {'name': f'{estimator_name}_real'}
            predict_real[estimator_name] = []
            predict_fake[estimator_name] = []
            for dataset, target, dataset_name in zip([self.real_x_test, self.fake_x_test],
                                                     [self.real_y_test, self.fake_y_test],
                                                     ['real', 'fake']):
                predict = real_estimator.predict(dataset)
                predict_real[estimator_name].append(predict)
                for score in self.scores:
                    row.update(score(target, predict, dataset_name))

            rows.append(row)

            # Using model trained by fake dataset to predict real and fake test datasets
            row = {'name': f'{estimator_name}_fake'}
            for dataset, target, dataset_name in zip([self.real_x_test, self.fake_x_test],
                                                     [self.real_y_test, self.fake_y_test],
                                                     ['real', 'fake']):
                predict = fake_estimator.predict(dataset)
                predict_fake[estimator_name].append(predict)
                for score in self.scores:
                    row.update(score(target, predict, dataset_name))

            rows.append(row)
        output = pd.DataFrame(rows)
        if self.validType == "classifer":
            rows = []
            for estimator_name in self.estimator_names:
                row = {'name': f'{estimator_name}'}
                for real, fake, name in zip(predict_real[estimator_name],
                                            predict_fake[estimator_name],
                                            ['real', 'fake']):
                    jac_sim = get_score("jaccard_similarity").score(real, fake)
                    row.update({name: jac_sim})
                rows.append(row)
            jaccard = pd.DataFrame(rows)
            self.logging.info(f"Jaccard Similarity of outputs predicted by models"
                              f"for real and fake testing datasets: \n {jaccard}\n")

        return output

    @abc.abstractmethod
    def build_estimators(self):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError
