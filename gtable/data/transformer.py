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
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.testing import ignore_warnings
import copy
import pickle
import numpy as np
import pandas as pd
import abc
from gtable.utils.misc import ClassRegistry


class Transformer(abc.ABC):
    def __init__(self, ctx, name):
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger

        self._name = name
        self.is_trained = False

    @property
    def name(self):
        return self._name

    def fit(self, data, discrete_colums=tuple(), metadata=None):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data, sigmas, **kwargs):
        raise NotImplementedError


_TRANSFORMER_REGISTRY = ClassRegistry(base_class=Transformer)
register_transformer = _TRANSFORMER_REGISTRY.register  # pylint: disable=invalid-name


@register_transformer(name="gmm")
class GMMTransformer(Transformer):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, ctx, name):
        super(GMMTransformer, self).__init__(ctx, name)
        self.meta = []

        self.dataframe = False
        self.column_names = None
        self.dtypes = None
        self.discrete_columns = None
        self.nums_trained_record = 0
        self.separated_embedding = ctx.config.separated_embedding
        self.disccret_embedding_name = ctx.config.discrete_embeddding
        self.continuous_embedding_name = ctx.config.continuous_embeddding

    def fit(self, data, discrete_colums=tuple(), metadata=None):
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        self.dtypes = data.infer_objects().dtypes
        self.column_names = data.columns
        self.discrete_columns = discrete_colums
        self.nums_trained_record = len(data)
        self.metadata = metadata

        if self.separated_embedding:
            for column in data.columns:
                column_data = data[[column]].values
                embedding_name = self.disccret_embedding_name if column in self.discrete_columns \
                    else self.continuous_embedding_name

                meta = build_embedding(embedding_name, column)
                meta.fit(column_data)

                self.output_info += meta.output_info
                self.output_dimensions += meta.output_dimensions
                self.meta.append(meta)
        else:
            meta = build_embedding(self.continuous_embedding_name, "WholeTable")
            meta.fit(data)
            self.output_info += meta.output_info
            self.output_dimensions += meta.output_dimensions
            self.meta.append(meta)
        self.is_trained = True

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if self.separated_embedding:
            values = []
            for meta in self.meta:
                if meta.column_name in data:
                    column_data = data[[meta.column_name]].values
                    if meta.column_name in self.discrete_columns:
                        values.append(meta.transform(column_data))
                    else:
                        values += meta.transform(column_data)

            return np.concatenate(values, axis=1).astype(float)
        else:
            return self.meta[0].transform(data)

    @staticmethod
    def nearest_value(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def rounding(self, fake, real, column_list):
        for i in column_list:
            fake[:, i] = np.array([self.nearest_value(real[:, i], x) for x in fake[:, i]])
        return fake

    def inverse_transform(self, data, sigmas, real=None, save=False, **kwargs):
        if self.separated_embedding:
            start = 0
            output = []
            column_names = []
            sigma = sigmas[start] if sigmas is not None else None
            for meta in self.meta:
                dimensions = meta.output_dimensions
                columns_data = data[:, start:start + dimensions]
                inverted = meta.inverse_transform(columns_data, sigma)
                output.append(inverted)
                column_names.append(meta.column_name)
                start += dimensions
            output = np.column_stack(output)
        else:
            column_names = self.column_names
            output = self.meta[0].inverse_transform(data, None)

        if real is not None:
            output = self.rounding(output, real, range(output.shape[1]))

        output = pd.DataFrame(output, columns=column_names).astype(self.dtypes)

        org_output = copy.copy(output)
        self.metadata.pop('colums_name')
        for attr in self.metadata.keys():
            index = self.metadata[attr]['label']
            org_output[attr] = org_output[attr].apply(lambda x: index[int(x)])

        if not self.dataframe:
            return output.values, org_output.values
        else:
            return output, org_output


@register_transformer(name="normal")
class NormalTransformer(Transformer):

    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.

    Args:
        n_cluster (int):
            Number of modes.
        epsilon (float):
            Epsilon value.
    """

    def __init__(self, ctx, name):
        super(NormalTransformer, self).__init__(ctx, name)
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
            'name': column,
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
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    def fit(self, data, discrete_columns=tuple(), metadata=None):
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        self.dtypes = data.infer_objects().dtypes
        self.meta = []
        for column in data.columns:
            column_data = data[[column]].values
            if column in discrete_columns:
                meta = self._fit_discrete(column, column_data)
            else:
                meta = self._fit_continuous(column, column_data)

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
        encoder = column_meta['encoder']
        return encoder.transform(data)

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self.meta:
            column_data = data[[meta['name']]].values
            if 'model' in meta:
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
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data, sigmas, **kwargs):  # (15000, 126)
        start = 0
        output = []
        column_names = []
        for meta in self.meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                sigma = sigmas[start] if sigmas is not None else None
                inverted = self._inverse_transform_continuous(meta, columns_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names).astype(self.dtypes)
        if not self.dataframe:
            output = output.values

        return output

    def save(self, path):
        with open(path + "/data_transform.pl", "wb") as f:
            pickle.dump(self, f)

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

    @classmethod
    def load(cls, path):
        with open(path + "/data_transform.pl", "rb") as f:
            return pickle.load(f)


def build_transformer(ctx):
    name = ctx.config.transformer_type
    transformer_class = _TRANSFORMER_REGISTRY.get(name.lower())
    if transformer_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return transformer_class(ctx, name)
