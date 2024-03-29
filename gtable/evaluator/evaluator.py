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

"""Evaluation related classes and functions."""
from gtable.evaluator.viz import get_plot
from typing import Tuple
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from dython.nominal import compute_associations, numerical_encoding
from gtable.evaluator.scores import get_score
from gtable.evaluator.task.class_task import ClassEvaluator
from gtable.evaluator.task.regr_task import RegrEvaluator
from gtable.utils.constants import NUMERICAL, CATEGORICAL, ORDINAL
from typing import Union
import pandas as pd
from typing import List
import numpy as np


class DataEvaluator:
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user
    to easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods
    of evaluate and the visual evaluation method.
    """

    def __init__(self, ctx, real_dataset, fake_dataset):
        self.context = ctx
        self.config = self.context.config
        self.logging = self.context.logger

        self.metadata = real_dataset.metadata

        self.real = pd.DataFrame(real_dataset.train_dataset, columns=real_dataset.name_columns)
        self.fake = pd.DataFrame(fake_dataset.train_dataset, columns=fake_dataset.name_columns)

        self.numerical_cols = [item['name'] for item in real_dataset.metadata['columns']
                               if item['type'] == NUMERICAL]

        self.categorial_cols = [item['name'] for item in real_dataset.metadata['columns']
                                if item['type'] == CATEGORICAL or item['type'] == ORDINAL]

        self.random_seed = self.config.seed

        self.num_samples = min(len(self.real), len(self.fake))

        self.real = self.real.sample(self.num_samples)
        self.fake = self.fake.sample(self.num_samples)
        assert len(self.real) == len(self.fake), f'len(real) != len(fake)'

        self.is_class_evaluator = False
        self.is_regr_evaluator = False

        if self.config.classify_tasks is not None:
            self.evaluator = ClassEvaluator(self.context,
                                            real=real_dataset,
                                            fake=fake_dataset,
                                            seed=self.random_seed)
            self.is_class_evaluator = True

        if self.config.regression_tasks is not None:
            self.evaluator = RegrEvaluator(self.context,
                                           real=real_dataset,
                                           fake=fake_dataset,
                                           seed=self.random_seed)
            self.is_regr_evaluator = True

        self.visual = [] if self.config.visual is None else self.config.visual

        # statistical methods for numerical attributes
        self.numerical_statistics = [] if self.config.numerical_statistics is None \
            else self.config.numerical_statistics

    def get_column_correlation_metrics(self, metrics=None) -> dict:
        """
        Calculate distance between correlation matrices with certain metric.
        Calculate the correlation/strength-of-association of features in data-set with both categorical and continuous features using: * Pearson's R for continuous-continuous cases * Correlation Ratio for categorical-continuous cases * Cramer's V or Theil's U for categorical-categorical cases
        :param metrics: metric to measure distance.
        Choose from [``euclidean``, ``mae``, ``rmse``, ``cosine``, pearsonr].
        :return: distance between the association matrices in the
        chosen evaluation metric. Default: Euclidean
        """
        if metrics is None:
            metrics = ['euclidean']

        return_metrics = {}

        real_corr = compute_associations(self.real,
                                         nominal_columns=self.categorial_cols,
                                         theil_u=True).values
        fake_corr = compute_associations(self.fake,
                                         nominal_columns=self.categorial_cols,
                                         theil_u=True).values
        for how in metrics:
            assert how in ['euclidean', 'mae', 'rmse', 'cosine', "pearsonr"]

            distance_func = get_score(how).score

            if how == "pearsonr":
                real_corr = real_corr[~np.eye(real_corr.shape[0], dtype=bool)].reshape(
                    real_corr.shape[0], -1).flatten()
                fake_corr = fake_corr[~np.eye(fake_corr.shape[0], dtype=bool)].reshape(
                    fake_corr.shape[0], -1).flatten()
                score, _ = distance_func(real_corr, fake_corr)
            else:
                score = distance_func(real_corr, fake_corr)

            return_metrics[f"{how.upper()}_Column_Correlation"] = score

        return return_metrics

    def get_copies(self, return_len: bool = False) -> Union[pd.DataFrame, int]:
        """
        Check whether any real values occur in the fake data.

        :param return_len: whether to return the length of the copied rows or not.
        :return: Dataframe containing the duplicates if return_len=False,
        else integer indicating the number of copied rows.
        """
        real_hashes = self.real.apply(lambda x: hash(tuple(x)), axis=1)
        fake_hashes = self.fake.apply(lambda x: hash(tuple(x)), axis=1)

        dup_idxs = fake_hashes.isin(real_hashes.values)
        dup_idxs = dup_idxs[dup_idxs == True].sort_index().index.tolist()

        self.logging.debug(f'Nr copied columns: {len(dup_idxs)}')
        copies = self.fake.loc[dup_idxs, :]

        if return_len:
            return len(copies)
        else:
            return copies

    def get_duplicates(self) -> dict:
        """
        Return duplicates within each dataset.
        """
        real_duplicates = self.real[self.real.duplicated(keep=False)]
        fake_duplicates = self.fake[self.fake.duplicated(keep=False)]

        # return {'Duplicate rows between sets (real/fake)':
        #             (len(real_duplicates), len(fake_duplicates))}
        return {'Duplicate_Real': len(real_duplicates),
                'Duplicate_Fake': len(fake_duplicates)}

    def pca_correlation(self, lingress=False):
        """
        Calculate the relation between PCA explained variance values.
        Due to some very large numbers,
        in recent implementation the MAPE(log) is used instead of
        regressions like Pearson's r.

        :param lingress: whether to use a linear regression, in this case Pearson's.
        :return: the correlation coefficient if lingress=True,
        otherwise 1 - MAPE(log(real), log(fake))
        """
        self.pca_r = PCA(n_components=5)
        self.pca_f = PCA(n_components=5)

        # real = self.real
        # fake = self.fake

        real = numerical_encoding(self.real, nominal_columns=self.categorial_cols)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorial_cols)

        self.pca_r.fit(real)
        self.pca_f.fit(fake)

        self.logging.debug(f'\nTop 5 PCA components: \n' +
                           pd.DataFrame({'real': self.pca_r.explained_variance_,
                                         'fake': self.pca_f.explained_variance_}).to_string())

        if lingress:
            corr, p, _ = get_score("pearsonr").score(self.pca_r.explained_variance_,
                                                     self.pca_f.explained_variance_)
            return corr
        else:
            pca_error = get_score("mape").score(self.pca_r.explained_variance_,
                                                self.pca_f.explained_variance_)
            return 1 - pca_error

    def statistical_evaluation(self) -> float:
        """
        Calculate the correlation coefficient between the basic properties of
        self.real and self.fake using Spearman's Rho. Spearman's is used because these
        values can differ a lot in magnitude, and Spearman's is more resilient to outliers.
        :return: correlation coefficient
        """

        def compute_numerical_statistics(lingress=True):
            total_metrics = pd.DataFrame()
            total_diff = None

            columns_metrics = None
            real_metrics = {}
            fake_metrics = {}
            for score in ['min', 'max', 'mean', 'median', 'std', 'var']:

                real_score = getattr(self.real[self.numerical_cols], score)
                fake_score = getattr(self.fake[self.numerical_cols], score)

                for idx, value in real_score().items():
                    real_metrics[f'{score}_{idx}'] = value

                for idx, value in fake_score().items():
                    fake_metrics[f'{score}_{idx}'] = value

                if score in self.numerical_statistics:
                    score_stats = pd.concat([real_score(), fake_score()], axis=1)
                    score_stats.columns = [f'{score}_real', f'{score}_fake']
                    columns_metrics = score_stats if columns_metrics is None \
                        else pd.concat([columns_metrics, score_stats], axis=1)
                    diff = (real_score() - fake_score()).abs().rename("sum_diff")
                    total_diff = diff if total_diff is None else total_diff + diff

            columns_metrics = pd.concat([columns_metrics, total_diff], axis=1) \
                if total_diff is not None else columns_metrics

            total_metrics['real'] = real_metrics.values()
            total_metrics['fake'] = fake_metrics.values()
            total_metrics.index = real_metrics.keys()

            if len(self.numerical_statistics) > 0:
                self.logging.info(f'Basic statistical information of each numerical attribute: '
                                  f'\n {columns_metrics.to_string()}\n')

            if lingress:
                corr, p = get_score("spearmanr").score(total_metrics['real'],
                                                       total_metrics['fake'])
                return corr
            else:
                error = get_score("mape").score(total_metrics['real'], total_metrics['fake'])
                return 1 - error

        def compute_categorial_statistics(lingress=True):
            total_metrics = pd.DataFrame()

            def get_metrics(desc):
                cols = desc.columns
                index = desc.index
                for _ in range(len(cols) - 1):
                    index = index.append(desc.index)

                melt = desc.melt()

                melt["name"] = melt['variable'] + "_" + index

                return melt[['name', 'value']].set_index('name').to_dict()['value']

            real_desc = self.real[self.categorial_cols].astype('category') \
                .describe().transpose().drop(["count"], axis=1)
            real_metrics = get_metrics(real_desc)

            fake_desc = self.fake[self.categorial_cols].astype('category') \
                .describe().transpose().drop(["count"], axis=1)
            fake_metrics = get_metrics(fake_desc)

            total_metrics['real'] = real_metrics.values()
            total_metrics['fake'] = fake_metrics.values()
            total_metrics.index = real_metrics.keys()

            if len(self.numerical_statistics) > 0:
                real_desc.columns = [f"{name}_real" for name in real_desc.columns]
                fake_desc.columns = [f"{name}_fake" for name in fake_desc.columns]

                _metrics = real_desc.join(fake_desc).sort_index(axis=1)
                _metrics['freq_diff'] = _metrics['freq_real'] - _metrics['freq_fake']

                self.logging.info(f'Basic statistical information of each categorial/ordinal '
                                  f'attribute: \n {_metrics.to_string()}\n')
            if lingress:
                corr, p = get_score("spearmanr").score(total_metrics['real'],
                                                       total_metrics['fake'])
                return corr
            else:
                error = get_score("mape").score(total_metrics['real'], total_metrics['fake'])
                return 1 - error

        return (compute_numerical_statistics() * len(self.numerical_cols) +
                compute_categorial_statistics() * len(self.categorial_cols)) / \
               (len(self.numerical_cols) + len(self.categorial_cols))

    def compute_distance(self, sample=None) -> dict:
        real = self.real.values
        fake = self.fake.values
        mask_d = np.zeros(len(self.metadata['columns']))

        if sample is None:
            sample = int(len(real) * 0.50)

        for id_, info in enumerate(self.metadata['columns']):
            if info['type'] in [CATEGORICAL, ORDINAL]:
                mask_d[id_] = 1
            else:
                mask_d[id_] = 0

        std = np.std(real, axis=0) + 1e-6

        dis_all = []
        for i in range(min(sample, len(real))):
            current = fake[i]
            distance_d = (real - current) * mask_d > 0
            distance_d = np.sum(distance_d, axis=1)

            distance_c = (real - current) * (1 - mask_d) / 2 / std
            distance_c = np.sum(distance_c ** 2, axis=1)
            distance = np.sqrt(np.min(distance_c + distance_d))
            dis_all.append(distance)

        return {'Record_Distance': np.mean(dis_all)}

    def convert_numerical(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Special function to convert dataset to a numerical representations while
        making sure they have identical columns. This is sometimes a problem with
        categorical columns with many values or very unbalanced values

        :return: Real and fake dataframe with categorical columns
        one-hot encoded and binary columns factorized.
        """
        real = numerical_encoding(self.real, nominal_columns=self.categorial_cols)

        columns = sorted(real.columns.tolist())
        real = real[columns]
        fake = numerical_encoding(self.fake, nominal_columns=self.categorial_cols)
        for col in columns:
            if col not in fake.columns.tolist():
                fake[col] = 0
        fake = fake[columns]
        return real, fake

    def compute_row_nearest_neighbor(self, num_samples: int = None) -> dict:
        """
        Calculate mean and standard deviation distances between `self.fake` and `self.real`.

        :param num_samples: Number of samples to take for evaluation.
        Compute time increases exponentially.
        :return: `(mean, std)` of these distances.
        """
        if num_samples is None:
            num_samples = len(self.real)
        real = numerical_encoding(self.real, nominal_columns=self.categorial_cols)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorial_cols)

        columns = sorted(real.columns.tolist())
        real = real[columns]

        for col in columns:
            if col not in fake.columns.tolist():
                fake[col] = 0
        fake = fake[columns]

        for column in self.numerical_cols:
            real[column] = (real[column] - real[column].mean()) / real[column].std()
            fake[column] = (fake[column] - fake[column].mean()) / fake[column].std()

        assert real.columns.tolist() == fake.columns.tolist()

        # (len_input, len_input)
        distances = cdist(real[:num_samples], fake[:num_samples])
        min_distances = np.min(distances, axis=1)
        min_mean = np.mean(min_distances)
        min_std = np.std(min_distances)

        return {'Neighbor_Mean': min_mean,
                'Neighbor_Std': min_std}

    def column_correlations(self):
        """
        Wrapper function around `metrics.column_correlation`.

        :return: Column correlations between ``self.real`` and ``self.fake``.
        """
        column_correlations = get_score("column_correlations")
        return column_correlations(self.real, self.fake, self.categorial_cols)

    def run(self, iteration=0):
        """
        Determine correlation between attributes from the real
        and fake dataset using a given metric.
        All metrics from scipy.stats are available.
        """

        for task in self.visual:
            self.logging.info(f"Plot {task} ...")
            get_plot(self.context, task)(self.real, self.fake,
                                         self.numerical_cols,
                                         self.categorial_cols)

        # warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        # pd.options.display.float_format = '{:,.4f}'.format

        miscellaneous = dict()
        miscellaneous.update(self.get_column_correlation_metrics(['rmse', 'mae', 'pearsonr']))
        miscellaneous.update(self.compute_distance())
        miscellaneous.update(self.compute_row_nearest_neighbor())
        miscellaneous.update(self.get_duplicates())

        all_results = {
            'Spearman_Statistics': self.statistical_evaluation(),
            'Column_Correlations': self.column_correlations(),
            'MAPE_PCA': self.pca_correlation(),
        }

        scores_metrics, eval_score = self.evaluator.run()

        # b = estimators_scores[['name', 'accuracy_real', 'f1_score_real']]
        # b.columns = scores_metrics.columns
        # scores_metrics = scores_metrics.append(b)

        if self.is_class_evaluator:
            self.logging.info(
                f'Metrics score of Classifier tasks:\n{scores_metrics.to_string()}\n')
            all_results.update({"MAPE_Class": eval_score})

        if self.is_regr_evaluator:
            self.logging.info(
                f'Metrics score of Regressor tasks:\n {scores_metrics.to_string()}\n')
            all_results.update({"RMSE_Regr": eval_score})

        all_results['Similarity_Score'] = np.mean(list(all_results.values()))

        for key, value in miscellaneous.items():
            scores_metrics[key] = value

        for key, value in all_results.items():
            scores_metrics[key] = value

        miscellaneous_df = pd.DataFrame({'Result': list(miscellaneous.values())},
                                        index=list(miscellaneous.keys()))
        self.logging.info(f'Miscellaneous results:\n{miscellaneous_df.to_string()}\n')

        all_results_df = pd.DataFrame({'Result': list(all_results.values())},
                                      index=list(all_results.keys()))
        self.logging.info(f'Summary Results:\n{all_results_df.to_string()}\n')

        return scores_metrics
