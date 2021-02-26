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

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, mean_squared_error, \
    euclidean_distances, mean_absolute_error, confusion_matrix, precision_score, recall_score

from sklearn.metrics.pairwise import cosine_similarity

from gtable.utils.misc import ClassRegistry
import abc
import pandas as pd
import numpy as np
import scipy.stats as ss
from dython.nominal import theils_u, cramers_v
from scipy.spatial.distance import cosine


class Scorer(abc.ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError()

    def __call__(self, y_true, y_pred, dataset):
        return {f"{self.name}_{dataset}": self.score(y_true, y_pred)}


_SCORERS_REGISTRY = ClassRegistry(base_class=Scorer)
register_scorer = _SCORERS_REGISTRY.register  # pylint: disable=invalid-name


@register_scorer(name="pearsonr")
class PearsonrScorer(Scorer):
    def __init__(self):
        super(PearsonrScorer, self).__init__("pearsonr")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return ss.pearsonr(y_true, y_pred)


@register_scorer(name="spearmanr")
class SpearmanrScorer(Scorer):
    def __init__(self):
        super(SpearmanrScorer, self).__init__("spearmanr")

    def score(self, y_true, y_pred):
        return ss.spearmanr(y_true, y_pred)


@register_scorer(name="confusion_matrix")
class ConfusionMatrixScorer(Scorer):
    def __init__(self):
        super(ConfusionMatrixScorer, self).__init__("confusion_matrix")

    def score(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)


@register_scorer(name="precision_score")
class PrecisionScorer(Scorer):
    def __init__(self):
        super(PrecisionScorer, self).__init__("precision_score")

    def score(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

@register_scorer(name="recall_score")
class RecallScorer(Scorer):
    def __init__(self):
        super(RecallScorer, self).__init__("recall_score")

    def score(self, y_true, y_pred):
        return recall_score(y_true, y_pred)


@register_scorer(name="roc_auc")
class ROCScorer(Scorer):
    def __init__(self):
        super(ROCScorer, self).__init__("roc_auc")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return roc_auc_score(y_true, y_pred)


@register_scorer(name="auc")
class AUCScorer(Scorer):
    def __init__(self):
        super(AUCScorer, self).__init__("auc")

    # P-R_Curve
    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)


@register_scorer(name="f1_score")
class F1Scorer(Scorer):
    def __init__(self):
        super(F1Scorer, self).__init__("f1_score")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        if np.unique(y_true).size <= 2:
            return f1_score(y_true, y_pred, average="binary")
        else:
            return f1_score(y_true, y_pred, average="micro")


@register_scorer(name="accuracy")
class AccuracyScorer(Scorer):
    def __init__(self):
        super(AccuracyScorer, self).__init__("accuracy")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return accuracy_score(y_true, y_pred)


# mean_squre_error
@register_scorer(name="mse")
class MSEScorer(Scorer):
    def __init__(self):
        super(MSEScorer, self).__init__("mse")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return mean_squared_error(y_true, y_pred)


# euclidean_distances
@register_scorer(name="euclidean")
class EUCDistanceScorer(Scorer):
    def __init__(self):
        super(EUCDistanceScorer, self).__init__("euclidean")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return euclidean_distances(y_true, y_pred)


# mean_absolute_error
@register_scorer(name="mae")
class MAEDistanceScorer(Scorer):
    def __init__(self):
        super(MAEDistanceScorer, self).__init__("mae")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return mean_absolute_error(y_true, y_pred)


# mean_absolute_percentage_error
@register_scorer(name="mape")
class MAPEDistanceScorer(Scorer):
    def __init__(self):
        super(MAPEDistanceScorer, self).__init__("mape")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
            Returns the mean absolute percentage error between y_true and y_pred.
            Throws ValueError if y_true contains zero values.

            :param y_true: NumPy.ndarray with the ground truth values.
            :param y_pred: NumPy.ndarray with the ground predicted values.
            :return: Mean absolute percentage error (float).
            """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true))


# Root Mean Square Error
@register_scorer(name="rmse")
class RMSEScorer(Scorer):
    def __init__(self):
        super(RMSEScorer, self).__init__("rmse")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        # return np.sqrt(mean_absolute_error(y_true, y_pred))
        return np.sqrt(mean_squared_error(y_true, y_pred))


# Root Mean Square Error
@register_scorer(name="cosine")
class CosineScorer(Scorer):
    def __init__(self):
        super(CosineScorer, self).__init__("cosine")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return cosine(y_true.reshape(-1), y_pred.reshape(-1))


@register_scorer(name="cosine_similarity")
class CosineSimilarityScorer(Scorer):
    def __init__(self):
        super(CosineSimilarityScorer, self).__init__("cosine_similarity")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        return cosine_similarity(y_true, y_pred)


@register_scorer(name="jaccard_similarity")
class JaccardSimilarityScorer(Scorer):
    def __init__(self):
        super(JaccardSimilarityScorer, self).__init__("jaccard_similarity")

    def score(self, y_true, y_pred):
        return jaccard_score(y_true, y_pred, average='micro')


@register_scorer(name="column_correlations")
class ColumnCorScorer(Scorer):
    def __init__(self):
        super(ColumnCorScorer, self).__init__("column_correlations")

    def score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

    def __call__(self, dataset_a, dataset_b, categorical_columns, theil_u=True, dataset=None):
        """
        Column-wise correlation calculation between ``dataset_a`` and ``dataset_b``.

        :param dataset_a: First DataFrame
        :param dataset_b: Second DataFrame
        :param categorical_columns: The columns containing categorical values
        :param theil_u: Whether to use Theil's U. If False, use Cramer's V.
        :return: Mean correlation between all columns.
        """
        if categorical_columns is None:
            categorical_columns = list()
        elif categorical_columns == 'all':
            categorical_columns = dataset_a.columns
        assert dataset_a.columns.tolist() == dataset_b.columns.tolist()
        corr = pd.DataFrame(columns=dataset_a.columns, index=['correlation'])

        for column in dataset_a.columns.tolist():
            if column in categorical_columns:
                if theil_u:
                    corr[column] = theils_u(dataset_a[column].sort_values(), dataset_b[column].sort_values())
                else:
                    corr[column] = cramers_v(dataset_a[column].sort_values(), dataset_b[column].sort_vaues())
            else:
                corr[column], _ = ss.pearsonr(dataset_a[column].sort_values(), dataset_b[column].sort_values())
        corr.fillna(value=np.nan, inplace=True)
        correlation = np.mean(corr.values.flatten())
        return correlation


def make_scores(names):
    if not isinstance(names, list):
        names = [names]
    return [get_score(name) for name in names]


def get_score(name):
    scorer_class = _SCORERS_REGISTRY.get(name.lower())
    if scorer_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return scorer_class()
