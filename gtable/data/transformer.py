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

from gtable.utils.embedding import build_embedding
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.testing import ignore_warnings
import copy
import pickle
import numpy as np
import pandas as pd
import abc
from gtable.utils.misc import ClassRegistry
from gtable.utils.constants import CATEGORICAL, NUMERICAL, ORDINAL
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import KBinsDiscretizer

"""
There are three kinds of data type:
1. Categorical: A categorical variable (sometimes called a nominal variable)
                is one that has two or more categories, but there is no intrinsic
                ordering to the categories.

2. Ordinal: An ordinal variable is similar to a categorical variable.  The
            difference between the two is that there is a clear ordering of
            the categories.

4. Numerical: An numerical variable is similar to an ordinal variable, except
              that the intervals between the values of the numerical variable
              are equally spaced.

"""


class Transformer(abc.ABC):
    def __init__(self, ctx, name, metadata):
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger

        self._name = name
        self.is_trained = False
        self.metadata = metadata

        self.meta = []
        self.output_info = []
        self.output_dimensions = 0

        self.unify_embedding = ctx.config.unify_embedding
        self.numerical_embeddding = ctx.config.numerical_embeddding
        self.categorial_embeddding = ctx.config.categorial_embeddding
        self.ordinal_embeddding = ctx.config.ordinal_embeddding
        self.embedding_combine = ctx.config.embedding_combine

    def get_embedding_name(self, item, data):
        atts_type = item['type']
        if self.unify_embedding is not None:
            embedding_name = self.unify_embedding
        elif atts_type == NUMERICAL:
            embedding_name = self.numerical_embeddding
            item['min'] = np.min(data)
            item['max'] = np.max(data)
            item['mean'] = np.mean(data)
        elif atts_type == CATEGORICAL:
            embedding_name = self.categorial_embeddding
            item['size'] = len(np.unique(data))
        elif atts_type == ORDINAL:
            embedding_name = self.ordinal_embeddding
            item['size'] = len(np.unique(data))
        else:
            embedding_name = self.unify_embedding

        return embedding_name

    @property
    def name(self):
        return self._name

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data, sigmas, **kwargs):
        raise NotImplementedError

    def covert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for info in self.meta:
            if info["name"] == column_name:
                break
            if len(info["output_info"]) == 1:  # is discrete column
                discrete_counter += 1
            column_id += 1

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(info["encoder"].transform([[value]])[0])
        }


_TRANSFORMER_REGISTRY = ClassRegistry(base_class=Transformer)
register_transformer = _TRANSFORMER_REGISTRY.register  # pylint: disable=invalid-name


@register_transformer(name="general")
class GeneralTransformer(Transformer):
    def __init__(self, ctx, name, metadata):
        super(GeneralTransformer, self).__init__(ctx, name, metadata)
        self.n_clusters = self.config.n_clusters
        self.epsilon = self.config.epsilon

    @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, column, data):
        gm = BayesianGaussianMixture(
            self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        gm.fit(data)
        components = gm.weights_ > self.epsilon
        num_components = components.sum()

        return {
            'name': column['name'],
            'type': column['type'],
            'model': gm,
            'components': components,
            'output_info': [(1, 'tanh'), (num_components, 'softmax')],
            'output_dimensions': 1 + num_components,
        }

    def _fit_discrete(self, column, data):
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(data)
        categories = len(ohe.categories_[0])

        return {
            'name': column['name'],
            'type': column['type'],
            'model': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories}

    def fit(self, data):

        for idx, item in enumerate(self.metadata['columns']):
            column_data = data[:, idx].reshape([-1, 1])
            if item['type'] == NUMERICAL:
                meta = self._fit_continuous(item, column_data)
            else:
                meta = self._fit_discrete(item, column_data)

            self.output_info += meta['output_info']
            self.output_dimensions += meta['output_dimensions']
            self.meta.append(meta)

        self.is_trained = True

    def _transform_continuous(self, column_meta, data):
        components = column_meta['components']
        model = column_meta['model']

        means = model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(model.covariances_).reshape((1, self.n_clusters))
        features = (data - means) / (4 * stds)

        probs = model.predict_proba(data)

        n_opts = components.sum()
        features = features[:, components]
        probs = probs[:, components]

        opt_sel = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            pp = probs[i] + 1e-6
            pp = pp / pp.sum()
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99)

        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        return [features, probs_onehot]

    def _transform_discrete(self, column_meta, data):
        model = column_meta['model']
        return model.transform(data)

    def transform(self, data):

        values = []
        for idx, meta in enumerate(self.meta):
            column_data = data[:, idx].reshape([-1, 1])
            if meta['type'] == NUMERICAL:
                values += self._transform_continuous(meta, column_data)
            else:
                values.append(self._transform_discrete(meta, column_data))

        return np.concatenate(values, axis=1).astype(float)

    def _inverse_transform_continuous(self, meta, data, sigma):
        model = meta['model']
        components = meta['components']

        u = data[:, 0]
        v = data[:, 1:]

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), self.n_clusters)) * -100
        v_t[:, components] = v
        v = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, meta, data):
        model = meta['model']
        return model.inverse_transform(data)

    def inverse_transform(self, data, sigmas, **kwargs):  # (15000, 126)
        start = 0
        output = []
        column_names = []
        for idx, meta in enumerate(self.meta):
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if meta['type'] == NUMERICAL:
                sigma = sigmas[start] if sigmas is not None else None
                inverted = self._inverse_transform_continuous(meta, columns_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        # output = pd.DataFrame(output, columns=column_names)

        return output

    def save(self, path):
        with open(path + "/data_transform.pl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path + "/data_transform.pl", "rb") as f:
            return pickle.load(f)


@register_transformer(name="gmm")
class GMMTransformer(Transformer):
    """Data Transformer.

    Model NUMERICAL columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, ctx, name, metadata):
        super(GMMTransformer, self).__init__(ctx, name, metadata)

        # self.dataframe = False
        # self.column_names = None

    def fit(self, data):
        for idx, item in enumerate(self.metadata['columns']):
            column_data = data[:, idx].reshape([-1, 1])
            embedding_name = self.get_embedding_name(item, column_data)
            meta = build_embedding(embedding_name, item)
            meta.fit(column_data)
            item['output_dimensions'] = meta.output_dimensions
            item['output_info'] = meta.output_info
            self.output_info += meta.output_info
            self.output_dimensions += meta.output_dimensions
            self.meta.append(meta)

        self.is_trained = True

    def transform(self, data):
        values = []
        for idx, meta in enumerate(self.meta):
            column_data = data[:, idx].reshape([-1, 1])

            transformed_data = meta.transform(column_data)

            if isinstance(transformed_data, list):
                values += transformed_data
            else:
                values.append(transformed_data)

            # if self.unify_embedding is not None:
            #     values.append(meta.transform(column_data))
            # elif meta.type == NUMERICAL:
            #     values += meta.transform(column_data)
            # elif meta.type == ORDINAL:
            #     values.append(meta.transform(column_data))
            # else:  # CATEGORICAL
            #     values.append(meta.transform(column_data))

        return np.concatenate(values, axis=1).astype(float)

    @staticmethod
    def nearest_value(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def rounding(self, fake, real, column_list):
        for i in column_list:
            fake[:, i] = np.array([self.nearest_value(real[:, i], x) for x in fake[:, i]])
        return fake

    def inverse_transform(self, data, sigmas, real=None, save=False, **kwargs):
        start = 0
        output = []
        column_names = []
        sigma = sigmas[start] if sigmas is not None else None
        for idx, meta in enumerate(self.meta):
            dimensions = meta.output_dimensions
            columns_data = data[:, start:start + dimensions]
            inverted = meta.inverse_transform(columns_data, sigma)
            output.append(inverted)
            column_names.append(meta.column_name)
            start += dimensions
        output = np.column_stack(output)

        if real is not None:
            output = self.rounding(output, real, range(output.shape[1]))

        # output = pd.DataFrame(output, columns=column_names)

        return output


class DiscretizeTransformer(Transformer):
    """Discretize NUMERICAL columns into several bins.

    Attributes:
        meta
        column_index
        discretizer(sklearn.preprocessing.KBinsDiscretizer)

    Transformation result is a int array.

    """

    def __init__(self, ctx, name):
        super(DiscretizeTransformer, self).__init__(ctx, name)
        # def __init__(self, n_bins):
        #     self.n_bins = n_bins
        self.n_bins = 10
        self.meta = None
        self.column_index = None
        self.discretizer = None

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.column_index = [
            index for index, info in enumerate(self.meta) if info['type'] == NUMERICAL]

        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode='ordinal', strategy='uniform')

        if not self.column_index:
            return

        self.discretizer.fit(data[:, self.column_index])

    def transform(self, data):
        """Transform data discretizing continous values.

        Args:
            data(pandas.DataFrame)

        Returns:
            numpy.ndarray

        """
        if self.column_index == []:
            return data.astype('int')

        data[:, self.column_index] = self.discretizer.transform(data[:, self.column_index])
        return data.astype('int')

    def inverse_transform(self, data):
        if self.column_index == []:
            return data

        data = data.astype('float32')
        data[:, self.column_index] = self.discretizer.inverse_transform(data[:, self.column_index])
        return data


@register_transformer(name="bgm")
class BGMTransformer(Transformer):
    """Model NUMERICAL columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.

    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, n_clusters=10, eps=0.005):
        """n_cluster is the upper bound of modes."""
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        model = []

        self.output_info = []
        self.output_dim = 0
        self.components = []
        for id_, info in enumerate(self.meta):
            if info['type'] == NUMERICAL:
                gm = BayesianGaussianMixture(
                    self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    n_init=1)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                comp = gm.weights_ > self.eps
                self.components.append(comp)

                self.output_info += [(1, 'tanh'), (np.sum(comp), 'softmax')]
                self.output_dim += 1 + np.sum(comp)
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == NUMERICAL:
                current = current.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (4 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))

                n_opts = sum(self.components[id_])
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(data), dtype='int')
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)

                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                values += [features, probs_onehot]
            else:
                col_t = np.zeros([len(data), info['size']])
                idx = list(map(info['i2s'].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == NUMERICAL:
                u = data[:, st]
                v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]

                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp

            else:
                current = data[:, st:st + info['size']]
                st += info['size']
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))

        return data_t


@register_transformer(name="tablegan")
class TableganTransformer(Transformer):
    def __init__(self, ctx, name, metadata):
        super(TableganTransformer, self).__init__(ctx, name, metadata)
        self.height = 0

        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= len(metadata['columns']):
                self.height = i
                break

    @staticmethod
    def get_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append({
                    "name": index,
                    "type": ORDINAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            else:
                meta.append({
                    "name": index,
                    "type": NUMERICAL,
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.minn = np.zeros(len(self.meta))
        self.maxx = np.zeros(len(self.meta))
        for i in range(len(self.meta)):
            if self.meta[i]['type'] == NUMERICAL:
                self.minn[i] = self.meta[i]['min'] - 1e-3
                self.maxx[i] = self.meta[i]['max'] + 1e-3
            else:
                self.minn[i] = -1e-3
                self.maxx[i] = self.meta[i]['size'] - 1 + 1e-3

    def transform(self, data):
        data = data.copy().astype('float32')
        data = (data - self.minn) / (self.maxx - self.minn) * 2 - 1
        if self.height * self.height > len(data[0]):
            padding = np.zeros((len(data), self.height * self.height - len(data[0])))
            data = np.concatenate([data, padding], axis=1)

        return data.reshape(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.reshape(-1, self.height * self.height)

        data_t = np.zeros([len(data), len(self.meta)])

        for id_, info in enumerate(self.meta):
            numerator = (data[:, id_].reshape([-1]) + 1)
            data_t[:, id_] = (numerator / 2) * (self.maxx[id_] - self.minn[id_]) + self.minn[id_]
            if info['type'] in [CATEGORICAL, ORDINAL]:
                data_t[:, id_] = np.round(data_t[:, id_])

        return data_t


def build_transformer(ctx, metadata):
    name = ctx.config.transformer_type
    transformer_class = _TRANSFORMER_REGISTRY.get(name.lower())
    if transformer_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return transformer_class(ctx, name, metadata)
