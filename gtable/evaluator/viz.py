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

from gtable.utils.misc import ClassRegistry
from dython.nominal import associations
from typing import Union, List
from sklearn.decomposition import PCA
from dython.nominal import numerical_encoding
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import abc
from os.path import join


class Visualization(abc.ABC):
    def __init__(self, ctx, name):
        self.context = ctx
        self.logging = ctx.logger
        self._name = name
        self.output_path = join(ctx.output, f"{ctx.app.lower()}_{self.name.lower()}.pdf")

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def plot(self,
             real: pd.DataFrame,
             fake: pd.DataFrame,
             numerical_cols: list,
             categorial_cols: list,
             **kwargs):
        raise NotImplementedError()

    def __call__(self, real, fake, numerical_cols, categorial_cols,
                 save: bool = True, **kwargs):
        plt = self.plot(real, fake, numerical_cols, categorial_cols)
        if save:
            plt.savefig(self.output_path, bbox_inches='tight')
        else:
            plt.show()


VISUALIZATION_REGISTRY = ClassRegistry(base_class=Visualization)
register_plot = VISUALIZATION_REGISTRY.register  # pylint: disable=invalid-name


@register_plot(name="pca")
class PlotPCA(Visualization):
    def __init__(self, ctx):
        super(PlotPCA, self).__init__(ctx, "pca")
        self.n_components = 2

    def plot(self,
             real: pd.DataFrame,
             fake: pd.DataFrame,
             numerical_cols: list,
             categorial_cols: list,
             **kwargs):
        """
        Plot the first two components of a PCA of real and fake data.
        """
        real = numerical_encoding(real, nominal_columns=categorial_cols)
        fake = numerical_encoding(fake, nominal_columns=categorial_cols)
        pca_r = PCA(n_components=self.n_components)
        pca_f = PCA(n_components=self.n_components)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('First two components of PCA', fontsize=16)
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        ax[0].set_title('Real data')
        ax[1].set_title('Fake data')
        return plt


@register_plot(name="variance")
class PlotVarCor(Visualization):
    def __init__(self, ctx):
        super(PlotVarCor, self).__init__(ctx, "variance")

    @staticmethod
    def get_variance(x: Union[pd.DataFrame, np.ndarray],
                     ax, cmap, **kwargs) -> np.ndarray:
        """
        Given a DataFrame, plot the correlation between columns.
        Function assumes all numeric continuous data. It masks
        the top half of the correlation matrix, since this holds
        the same values.

        Decomissioned for use of the dython associations function.

        :param x: Dataframe to plot data from
        :param ax: Axis on which to plot the correlations
        :param cmap: return correlation matrix after plotting
        """
        if isinstance(x, pd.DataFrame):
            corr = x.corr().values
        elif isinstance(x, np.ndarray):
            corr = np.corrcoef(x, rowvar=False)
        else:
            raise ValueError('Unknown datatype given. Make sure a Pandas '
                             'DataFrame or Numpy Array is passed for x.')

        sns.set(style="white")
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, **kwargs)
        return corr

    def plot(self,
             real: pd.DataFrame,
             fake: pd.DataFrame,
             numerical_cols: list,
             categorial_cols: list,
             **kwargs):
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Variance correlation of real and fake dataset\n', fontsize=16)

        self.get_variance(real, ax=ax[0], cmap=cmap)
        self.get_variance(fake, ax=ax[1], cmap=cmap)

        titles = ['Real', 'Fake']
        for i, label in enumerate(titles):
            title_font = {'size': '18'}
            ax[i].set_title(label, **title_font)
        plt.tight_layout()
        return plt


@register_plot(name="mean_std")
class PlotMeanStd(Visualization):
    def __init__(self, ctx):
        super(PlotMeanStd, self).__init__(ctx, "mean_std")

    def plot(self,
             real: pd.DataFrame,
             fake: pd.DataFrame,
             numerical_cols: list,
             categorial_cols: list,
             **kwargs):
        """
          Plot the means and standard deviations of each dataset.

          :param real: DataFrame containing the real data
          :param fake: DataFrame containing the fake data
          :param numerical_cols
          :param categorial_cols
          """

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Absolute Log Mean and STDs of numeric data\n', fontsize=16)

        ax[0].grid(True)
        ax[1].grid(True)
        real = numerical_encoding(real, nominal_columns=categorial_cols)
        fake = numerical_encoding(fake, nominal_columns=categorial_cols)
        real_mean = np.log(np.add(abs(real.mean()).values, 1e-5))
        fake_mean = np.log(np.add(abs(fake.mean()).values, 1e-5))
        min_mean = min(real_mean) - 1
        max_mean = max(real_mean) + 1
        line = np.arange(min_mean, max_mean)
        sns.lineplot(x=line, y=line, ax=ax[0])
        sns.scatterplot(x=real_mean,
                        y=fake_mean,
                        ax=ax[0])
        ax[0].set_title('Means of real and fake data')
        ax[0].set_xlabel('real data mean (log)')
        ax[0].set_ylabel('fake data mean (log)')

        real_std = np.log(np.add(real.std().values, 1e-5))
        fake_std = np.log(np.add(fake.std().values, 1e-5))
        min_std = min(real_std) - 1
        max_std = max(real_std) + 1
        line = np.arange(min_std, max_std)
        sns.lineplot(x=line, y=line, ax=ax[1])
        sns.scatterplot(x=real_std,
                        y=fake_std,
                        ax=ax[1])
        ax[1].set_title('Stds of real and fake data')
        ax[1].set_xlabel('real data std (log)')
        ax[1].set_ylabel('fake data std (log)')
        return plt

    def plot_mean_std_comparison(self, evaluators: List):
        """
        Plot comparison between the means and standard
        deviations from each evaluator in evaluators.

        :param evaluators: list of TableEvaluator objects
        that are to be evaluated.
        """
        nr_plots = len(evaluators)
        fig, ax = plt.subplots(2, nr_plots, figsize=(4 * nr_plots, 7))
        flat_ax = ax.flatten()
        for i in range(nr_plots):
            self.plot(evaluators[i].real, evaluators[i].fake, list(), list(), ax=ax[:, i])

        titles = [e.name if e is not None else idx for idx, e in enumerate(evaluators)]
        for i, label in enumerate(titles):
            title_font = {'size': '24'}
            flat_ax[i].set_title(label, **title_font)
        plt.tight_layout()


@register_plot(name="correlation")
class PlotCorrelation(Visualization):
    def __init__(self, ctx):
        super(PlotCorrelation, self).__init__(ctx, "correlation")

    @staticmethod
    def plot_correlation_comparison(evaluators: List, annot=False):
        """
        Plot the correlation differences of multiple TableEvaluator objects.

        :param evaluators: list of TableEvaluator objects
        :param boolean annot: Whether to annotate the plots with numbers.
        """
        nr_plots = len(evaluators) + 1
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots(2, nr_plots, figsize=(4 * nr_plots, 7))
        flat_ax = ax.flatten()
        flat_ax[nr_plots + 1].clear()
        fake_corr = []
        real_corr = associations(evaluators[0].real,
                                 nominal_columns=evaluators[0].categorical_columns,
                                 plot=False,
                                 theil_u=True,
                                 mark_columns=True,
                                 annot=False,
                                 cmap=cmap,
                                 cbar=False,
                                 ax=flat_ax[0])['corr']
        for i in range(1, nr_plots):
            cbar = True if i % (nr_plots - 1) == 0 else False
            fake_corr.append(
                associations(evaluators[i - 1].fake,
                             nominal_columns=evaluators[0].categorical_columns,
                             plot=False,
                             theil_u=True,
                             mark_columns=True,
                             annot=False,
                             cmap=cmap,
                             cbar=cbar,
                             ax=flat_ax[i])['corr']
            )
            if i % (nr_plots - 1) == 0:
                cbar = flat_ax[i].collections[0].colorbar
                cbar.ax.tick_params(labelsize=20)

        for i in range(1, nr_plots):
            cbar = True if i % (nr_plots - 1) == 0 else False
            diff = abs(real_corr - fake_corr[i - 1])
            sns.set(style="white")
            az = sns.heatmap(diff,
                             ax=flat_ax[i + nr_plots],
                             cmap=cmap,
                             vmax=.3,
                             square=True,
                             annot=annot,
                             center=0,
                             linewidths=0,
                             cbar=cbar,
                             fmt='.2f')
            if i % (nr_plots - 1) == 0:
                cbar = az.collections[0].colorbar
                cbar.ax.tick_params(labelsize=20)
        titles = ['Real'] + [e.name if e.name is not None
                             else idx for idx, e in enumerate(evaluators)]
        for i, label in enumerate(titles):
            flat_ax[i].set_yticklabels([])
            flat_ax[i].set_xticklabels([])
            flat_ax[i + nr_plots].set_yticklabels([])
            flat_ax[i + nr_plots].set_xticklabels([])
            title_font = {'size': '28'}
            flat_ax[i].set_title(label, **title_font)
        plt.tight_layout()

    def plot(self,
             real: pd.DataFrame,
             fake: pd.DataFrame,
             numerical_cols: list,
             categorial_cols: list,
             **kwargs):
        """
         Plot the association matrices for the `real` dataframe, `fake` dataframe and plot
         the difference between them. Has support for continuous and Categorical
         (Male, Female) data types. All Object and Category dtypes are considered
         to be Categorical columns if `dis_cols` is not passed.

         - Continuous - Continuous: Uses Pearson's correlation coefficient
         - Continuous - Categorical: Uses so called correlation ratio
                        (https://en.wikipedia.org/wiki/Correlation_ratio) for
                        both continuous - categorical and categorical - continuous.
         - Categorical - Categorical: Uses Theil's U, an asymmetric correlation
                        metric for Categorical associations

         :param real: DataFrame with real data
         :param fake: DataFrame with synthetic data
         :param numerical_cols: List of numerical columns
         :param categorial_cols: List of Categorical columns
         """

        plot_diff: bool = True  # Plot difference if True, else not

        # Whether to annotate the plot with numbers
        # indicating the associations.
        annot = False

        assert isinstance(real, pd.DataFrame), f'`real` parameters must be a Pandas DataFrame'
        assert isinstance(fake, pd.DataFrame), f'`fake` parameters must be a Pandas DataFrame'
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        if categorial_cols is None:
            categorial_cols = real.select_dtypes(['object', 'category'])
        if plot_diff:
            fig, ax = plt.subplots(1, 3, figsize=(24, 7))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))

        real_corr = associations(real, nominal_columns=categorial_cols, plot=False, theil_u=True,
                                 mark_columns=True, annot=annot, ax=ax[0], cmap=cmap)['corr']
        fake_corr = associations(fake, nominal_columns=categorial_cols, plot=False, theil_u=True,
                                 mark_columns=True, annot=annot, ax=ax[1], cmap=cmap)['corr']

        if plot_diff:
            diff = abs(real_corr - fake_corr)
            sns.set(style="white")
            sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,
                        linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

        titles = ['Real', 'Fake', 'Difference'] if plot_diff else ['Real', 'Fake']
        for i, label in enumerate(titles):
            title_font = {'size': '18'}
            ax[i].set_title(label, **title_font)
        plt.tight_layout()
        return plt


@register_plot(name="cumsums")
class PlotCumSums(Visualization):
    def __init__(self, ctx):
        super(PlotCumSums, self).__init__(ctx, "cumsums")

    @staticmethod
    def cdf(data_r, data_f, xlabel: str = 'Values',
            ylabel: str = 'Cumulative Sum', ax=None):
        """
        Plot continous density function on optionally given ax.
        If no ax, cdf is plotted and shown.

        :param data_r: Series with real data
        :param data_f: Series with fake data
        :param xlabel: Label to put on the x-axis
        :param ylabel: Label to put on the y-axis
        :param ax: The axis to plot on. If ax=None, a new figure is created.
        """
        x1 = np.sort(data_r)
        x2 = np.sort(data_f)
        y = np.arange(1, len(data_r) + 1) / len(data_r)

        ax = ax if ax else plt.subplots()[1]

        axis_font = {'size': '14'}
        ax.set_xlabel(xlabel, **axis_font)
        ax.set_ylabel(ylabel, **axis_font)

        ax.grid()
        ax.plot(x1, y, marker='o', linestyle='none', label='Real', ms=8)
        ax.plot(x2, y, marker='o', linestyle='none', label='Fake', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        # If labels are strings, rotate them vertical
        if isinstance(data_r, pd.Series) and data_r.dtypes == 'object':
            ax.set_xticklabels(data_r.value_counts().sort_index().index,
                               rotation='vertical')

        if ax is None:
            plt.show()

    def plot(self,
             real: pd.DataFrame,
             fake: pd.DataFrame,
             numerical_cols: list,
             categorial_cols: list,
             **kwargs):
        """
        Plot the cumulative sums for all columns in the real and fake dataset. Height of each
        row scales with the length of the labels. Each plot contains the
        values of a real columns and the corresponding fake column.
        """
        nr_cols = 4
        nr_charts = len(real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(real.columns):
            r = real[col]
            f = fake.iloc[:, real.columns.tolist().index(col)]
            self.cdf(r, f, col, 'Cumsum', ax=axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        return plt


@register_plot(name="distributions")
class PlotDistribution(Visualization):
    def __init__(self, ctx):
        super(PlotDistribution, self).__init__(ctx, "distributions")

    def plot(self,
             real: pd.DataFrame,
             fake: pd.DataFrame,
             numerical_cols: list,
             categorial_cols: list,
             **kwargs):
        """
        Plot the distribution plots for all columns in the real and fake dataset.
        Height of each row of plots scales with the length of the labels. Each plot
        contains the values of a real columns and the corresponding fake column.
        """
        nr_cols = 3
        nr_charts = len(real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not real.select_dtypes(include=['object']).empty:
            lengths = []
            for d in real.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in real[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Distribution per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(real.columns):
            if col not in categorial_cols:
                sns.distplot(real[col], ax=axes[i], label='Real')
                sns.distplot(fake[col], ax=axes[i], color='darkorange', label='Fake')
                axes[i].legend()
            else:
                real = real.copy()
                fake = fake.copy()
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
        return plt


def get_plot(ctx, name):
    plot_class = VISUALIZATION_REGISTRY.get(name.lower())
    if plot_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return plot_class(ctx)
