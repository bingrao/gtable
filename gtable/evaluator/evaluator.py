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

"""Evaluation related classes and functions."""
from gtable.evaluator.viz import *
from typing import Tuple
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from dython.nominal import compute_associations, numerical_encoding
from gtable.evaluator.scores import get_score
from gtable.evaluator.task.class_task import ClassEvaluator
from gtable.evaluator.task.regr_task import RegrEvaluator
import pandas as pd


class DataEvaluator:
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user
    to easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods
    of evaluate and the visual evaluation method.
    """

    def __init__(self, ctx, real, fake):
        self.context = ctx
        self.config = self.context.config
        self.logging = self.context.logger

        self.real = real.dataset.copy()  # Pandas Dataframe
        self.fake = fake.dataset.copy()  # Pandas Dataframe
        self.numerical_cols = real.numerical_cols
        self.categorial_cols = fake.categorial_cols

        # the metric to use for evaluation linear relations. Pearson's r by default,
        # but supports all models in scipy.stats
        # self.comparison_metric = getattr(stats, metric)
        self.comparison_metric = get_score("pearsonr")

        self.random_seed = self.config.seed

        if self.config.n_samples is None:
            self.n_samples = min(len(self.real), len(self.fake))
        elif len(self.fake) >= self.config.n_samples and len(self.real) >= self.config.n_samples:
            self.n_samples = self.config.n_samples
        else:
            raise Exception(f'Make sure n_samples < len(fake/real). len(real): '
                            f'{len(self.real)}, len(fake): {len(self.fake)}')

        self.real = self.real.sample(self.n_samples)
        self.fake = self.fake.sample(self.n_samples)
        assert len(self.real) == len(self.fake), f'len(real) != len(fake)'

        self.is_class_evaluator = False
        self.is_regr_evaluator = False

        if self.config.classify_tasks is not None:
            self.evaluator = ClassEvaluator(self.context, real=real, fake=fake,
                                            numerical_columns=self.numerical_cols,
                                            categorical_columns=self.categorial_cols,
                                            seed=self.random_seed)
            self.is_class_evaluator = True

        if self.config.regression_tasks is not None:
            self.evaluator = RegrEvaluator(self.context, real=real, fake=fake,
                                           numerical_columns=self.numerical_cols,
                                           categorical_columns=self.categorial_cols,
                                           seed=self.random_seed)
            self.is_regr_evaluator = True

        self.visual = [] if self.config.visual is None else self.config.visual
        self.str2plotVisual = {"mean_std": self.plot_mean_std,
                               "cumsums": self.plot_cumsums,
                               "distributions": self.plot_distributions,
                               "correlation": self.plot_correlation_difference,
                               "pca": self.plot_pca}

        # statistical methods for numerical attributes
        self.continuous_statistics = [] if self.config.continuous_statistics is None \
            else self.config.continuous_statistics

        # statistical methods for categorial attributes
        self.discrete_statistics = [] if self.config.discrete_statistics is None \
            else self.config.discrete_statistics

    def plot_mean_std(self):
        """
        Class wrapper function for plotting the mean and std using `viz.plot_mean_std`.
        """
        plot_mean_std(self.real, self.fake)

    def plot_cumsums(self, nr_cols=4):
        """
        Plot the cumulative sums for all columns in the real and fake dataset. Height of each
        row scales with the length of the labels. Each plot contains the
        values of a real columns and the corresponding fake column.
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            r = self.real[col]
            f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
            cdf(r, f, col, 'Cumsum', ax=axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show()

    def plot_distributions(self, nr_cols=3):
        """
        Plot the distribution plots for all columns in the real and fake dataset.
        Height of each row of plots scales with the length of the labels. Each plot
        contains the values of a real columns and the corresponding fake column.
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Distribution per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            if col not in self.categorial_cols:
                sns.distplot(self.real[col], ax=axes[i], label='Real')
                sns.distplot(self.fake[col], ax=axes[i], color='darkorange', label='Fake')
                axes[i].legend()
            else:
                real = self.real.copy()
                fake = self.fake.copy()
                real['kind'] = 'Real'
                fake['kind'] = 'Fake'
                concat = pd.concat([fake, real])
                palette = sns.color_palette(
                    [(0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                     (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
                x, y, hue = col, "proportion", "kind"
                ax = (concat[x]
                      .groupby(concat[hue])
                      .value_counts(normalize=True)
                      .rename(y)
                      .reset_index()
                      .pipe((sns.barplot, "data"), x=x, y=y, hue=hue, ax=axes[i],
                            saturation=0.8, palette=palette))
                ax.set_xticklabels(axes[i].get_xticklabels(), rotation='vertical')
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show()

    def plot_correlation_difference(self, plot_diff=True, **kwargs):
        """
        Plot the association matrices for each table and, if chosen, the difference between them.

        :param plot_diff: whether to plot the difference
        :param kwargs: kwargs for sns.heatmap
        """
        plot_correlation_difference(self.real, self.fake,
                                    cat_cols=self.categorial_cols,
                                    plot_diff=plot_diff, **kwargs)

    def correlation_distance(self, how: str = 'euclidean') -> float:
        """
        Calculate distance between correlation matrices with certain metric.

        :param how: metric to measure distance. Choose from [``euclidean``, ``mae``, ``rmse``].
        :return: distance between the association matrices in the
        chosen evaluation metric. Default: Euclidean
        """
        from scipy.spatial.distance import cosine
        if how == 'euclidean':
            distance_func = get_score("euclidean").score
        elif how == 'mae':
            distance_func = get_score("mae").score
        elif how == 'rmse':
            distance_func = get_score("rmse").score
        elif how == 'cosine':
            def custom_cosine(a, b):
                return cosine(a.reshape(-1), b.reshape(-1))

            distance_func = custom_cosine
        else:
            raise ValueError(f'`how` parameter must be in [euclidean, mae, rmse]')

        real_corr = compute_associations(self.real,
                                         nominal_columns=self.categorial_cols, theil_u=True)
        fake_corr = compute_associations(self.fake,
                                         nominal_columns=self.categorial_cols, theil_u=True)

        return distance_func(real_corr.values, fake_corr.values)

    def plot_pca(self):
        """
        Plot the first two components of a PCA of real and fake data.
        """
        real = numerical_encoding(self.real, nominal_columns=self.categorial_cols)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorial_cols)
        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('First two components of PCA', fontsize=16)
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        ax[0].set_title('Real data')
        ax[1].set_title('Fake data')
        plt.show()

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

    def get_duplicates(self, return_values: bool = False) -> Tuple[Union[pd.DataFrame, int],
                                                                   Union[pd.DataFrame, int]]:
        """
        Return duplicates within each dataset.

        :param return_values: whether to return the duplicate values in the datasets.
        If false, the lengths are returned.
        :return: dataframe with duplicates or the length
        of those dataframes if return_values=False.
        """
        real_duplicates = self.real[self.real.duplicated(keep=False)]
        fake_duplicates = self.fake[self.fake.duplicated(keep=False)]
        if return_values:
            return real_duplicates, fake_duplicates
        else:
            return len(real_duplicates), len(fake_duplicates)

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

        real = self.real
        fake = self.fake

        real = numerical_encoding(real, nominal_columns=self.categorial_cols)
        fake = numerical_encoding(fake, nominal_columns=self.categorial_cols)

        self.pca_r.fit(real)
        self.pca_f.fit(fake)

        self.logging.debug(f'\nTop 5 PCA components: \n' +
                           pd.DataFrame({'real': self.pca_r.explained_variance_,
                                         'fake': self.pca_f.explained_variance_}).to_string())

        if lingress:
            corr, p, _ = self.comparison_metric.score(self.pca_r.explained_variance_,
                                                      self.pca_f.explained_variance_)
            return corr
        else:
            pca_error = get_score("mape").score(self.pca_r.explained_variance_,
                                                self.pca_f.explained_variance_)
            return 1 - pca_error

    def numerical_statistics(self):
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

            if score in self.continuous_statistics:
                score_stats = pd.concat([real_score(), fake_score()], axis=1)
                score_stats.columns = [f'{score}_real', f'{score}_fake']
                columns_metrics = score_stats if columns_metrics is None \
                    else pd.concat([columns_metrics, score_stats], axis=1)
                diff = (real_score() - fake_score()).abs().rename("sum_diff")
                total_diff = diff if total_diff is None else total_diff + diff

        columns_metrics = pd.concat([columns_metrics, total_diff], axis=1)

        total_metrics['real'] = real_metrics.values()
        total_metrics['fake'] = fake_metrics.values()
        total_metrics.index = real_metrics.keys()

        if len(self.continuous_statistics) > 0:
            self.logging.info(f'Basic statistical information of each numerical attribute: '
                              f'\n {columns_metrics.to_string()}\n')

        corr, p = get_score("spearmanr").score(total_metrics['real'], total_metrics['fake'])
        return corr

    def categorial_statistics(self):
        total_metrics = pd.DataFrame()
        columns_metrics = None
        real_metrics = {}
        fake_metrics = {}
        for score in ['min', 'max', 'mean', 'median', 'std']:

            real_score = getattr(self.real[self.categorial_cols], score)
            fake_score = getattr(self.fake[self.categorial_cols], score)

            for idx, value in real_score().items():
                real_metrics[f'{score}_{idx}'] = value

            for idx, value in fake_score().items():
                fake_metrics[f'{score}_{idx}'] = value

            if score in self.discrete_statistics:
                score_stats = pd.concat([real_score(), fake_score()], axis=1)
                score_stats.columns = [f'{score}_real', f'{score}_fake']
                columns_metrics = score_stats if columns_metrics is None \
                    else pd.concat([columns_metrics, score_stats], axis=1)

        total_metrics['real'] = real_metrics.values()
        total_metrics['fake'] = fake_metrics.values()
        total_metrics.index = real_metrics.keys()

        if len(self.discrete_statistics) > 0:
            self.logging.info(f'Basic statistical information of each categorial attribute: '
                              f'\n {columns_metrics.to_string()}\n')

        corr, p = get_score("spearmanr").score(total_metrics['real'], total_metrics['fake'])
        return corr

    def statistical_evaluation(self) -> float:
        """
        Calculate the correlation coefficient between the basic properties of
        self.real and self.fake using Spearman's Rho. Spearman's is used because these
        values can differ a lot in magnitude, and Spearman's is more resilient to outliers.
        :return: correlation coefficient
        """
        # return self.numerical_statistics() + self.categorial_statistics()

        return self.numerical_statistics()

    def correlation_correlation(self) -> float:
        """
        Calculate the correlation coefficient between the association
        matrices of self.real and self.fake using self.comparison_metric

        :return: The correlation coefficient
        """
        total_metrics = pd.DataFrame()
        for ds_name in ['real', 'fake']:
            ds = getattr(self, ds_name)
            corr_df = compute_associations(ds, nominal_columns=self.categorial_cols, theil_u=True)
            values = corr_df.values
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(values.shape[0], -1)
            total_metrics[ds_name] = values.flatten()

        self.correlation_correlations = total_metrics
        corr, p = self.comparison_metric.score(total_metrics['real'], total_metrics['fake'])

        self.logging.debug(f'\nColumn correlation between datasets:\n{total_metrics.to_string()}')

        return corr

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

    def row_distance(self, n_samples: int = None) -> Tuple[float, float]:
        """
        Calculate mean and standard deviation distances between `self.fake` and `self.real`.

        :param n_samples: Number of samples to take for evaluation.
        Compute time increases exponentially.
        :return: `(mean, std)` of these distances.
        """
        if n_samples is None:
            n_samples = len(self.real)
        real = numerical_encoding(self.real, nominal_columns=self.categorial_cols)
        fake = numerical_encoding(self.fake, nominal_columns=self.categorial_cols)

        columns = sorted(real.columns.tolist())
        real = real[columns]

        for col in columns:
            if col not in fake.columns.tolist():
                fake[col] = 0
        fake = fake[columns]

        for column in real.columns.tolist():
            if len(real[column].unique()) > 2:
                real[column] = (real[column] - real[column].mean()) / real[column].std()
                fake[column] = (fake[column] - fake[column].mean()) / fake[column].std()
        assert real.columns.tolist() == fake.columns.tolist()

        distances = cdist(real[:n_samples], fake[:n_samples])
        min_distances = np.min(distances, axis=1)
        min_mean = np.mean(min_distances)
        min_std = np.std(min_distances)
        return min_mean, min_std

    def column_correlations(self):
        """
        Wrapper function around `metrics.column_correlation`.

        :return: Column correlations between ``self.real`` and ``self.fake``.
        """
        column_correlations = get_score("column_correlations")
        return column_correlations(self.real, self.fake, self.categorial_cols)

    def run(self):
        """
        Determine correlation between attributes from the real
        and fake dataset using a given metric.
        All metrics from scipy.stats are available.
        """

        for task in self.visual:
            self.str2plotVisual[task]()

        # warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        # pd.options.display.float_format = '{:,.4f}'.format

        basic_statistical = self.statistical_evaluation()
        correlation_correlation = self.correlation_correlation()
        column_correlation = self.column_correlations()

        # pca_variance = self.pca_correlation()

        nearest_neighbor = self.row_distance()

        miscellaneous = {}
        miscellaneous['RMSE Column Correlation Distance'] = self.correlation_distance(how='rmse')
        miscellaneous['MAE Column Correlation Distance'] = self.correlation_distance(how='mae')
        miscellaneous['Mean Nearest Neighbor'] = nearest_neighbor[0]
        miscellaneous['std Nearest Neighbor'] = nearest_neighbor[1]
        miscellaneous['Duplicate rows between sets (real/fake)'] = self.get_duplicates()
        miscellaneous_df = pd.DataFrame({'Result': list(miscellaneous.values())},
                                        index=list(miscellaneous.keys()))

        all_results = {
            'Spearman correlation of basic statistics': basic_statistical,
            'Pearsonr correlation of column correlations': correlation_correlation,
            'Mean of correlation between real/fake columns': column_correlation,
            # 'MAPE 5 PCA components': pca_variance,
        }
        if self.is_class_evaluator:
            all_results.update({"MAPE on F1_Scores of classification tasks": self.evaluator.run()})

        if self.is_regr_evaluator:
            all_results.update({"Correlation on RMSE of regression tasks": self.evaluator.run()})

        total_result = np.mean(list(all_results.values()))
        all_results['Similarity Score [mean of all metrics]'] = total_result
        all_results_df = pd.DataFrame({'Result': list(all_results.values())},
                                      index=list(all_results.keys()))

        if len(self.continuous_statistics) > 0:
            self.logging.info(f'Miscellaneous results:\n{miscellaneous_df.to_string()}\n')

        if len(self.continuous_statistics) > 0:
            self.logging.info(f'Summary Results:\n{all_results_df.to_string()}\n')
