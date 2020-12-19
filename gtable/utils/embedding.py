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

from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import abc
from unknot.utils.misc import ClassRegistry
import numpy as np


class Embedding(abc.ABC):
    def __init__(self, name, column):
        self._name = name
        self._column_name = column
        self.model = self.build_model()

    @property
    def column_name(self):
        return self._column_name

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(self, data):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, data):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, data, sigma):
        raise NotImplementedError()


_EMBEDDING_REGISTRY = ClassRegistry(base_class=Embedding)
register_embedding = _EMBEDDING_REGISTRY.register  # pylint: disable=invalid-name


@register_embedding(name="bayesian_gaussian_norm")
class BayesianGaussianEmbedding(Embedding):
    def __init__(self, column):
        self.components = None
        self.output_info = None
        self.output_dimensions = 1
        self.n_clusters = 10
        self.epsilon = 0.005
        super(BayesianGaussianEmbedding, self).__init__("bayesian_gaussian_norm", column)

    def build_model(self):
        return BayesianGaussianMixture(
            self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1)

    def fit(self, data):
        self.model.fit(data)
        self.components = self.model.weights_ > self.epsilon
        num_components = self.components.sum()
        self.output_info = [(1, 'tanh'), (num_components, 'softmax')]
        self.output_dimensions = 1 + num_components

    def transform(self, data):
        means = self.model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(self.model.covariances_).reshape((1, self.n_clusters))
        features = (data - means) / (4 * stds)

        probs = self.model.predict_proba(data)

        n_opts = self.components.sum()
        features = features[:, self.components]
        probs = probs[:, self.components]

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
        # return features

    def inverse_transform(self, data, sigma):
        u = data[:, 0]  # scalar value
        v = data[:, 1:] # one-hot encoding

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), self.n_clusters)) * -100
        v_t[:, self.components] = v
        v = v_t
        means = self.model.means_.reshape([-1])
        stds = np.sqrt(self.model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * 4 * std_t + mean_t

        return column


@register_embedding(name="standard_norm")
class MinMaxScalarEmbedding(Embedding):
    def __init__(self, column):
        super(MinMaxScalarEmbedding, self).__init__("standard_norm", column)

    def build_model(self):
        return MinMaxScaler(feature_range=(-1, 1))

    def fit(self, data):
        self.model.fit(data)
        self.components = data.shape[1]
        self.output_info = [(self.components, 'tanh')]
        self.output_dimensions = self.components

    def transform(self, data):
        return self.model.transform(data)

    def inverse_transform(self, data, sigma):
        return self.model.inverse_transform(data)


@register_embedding(name="one_hot_vector")
class OneHotEmbedding(Embedding):
    def __init__(self, column):
        super(OneHotEmbedding, self).__init__("one_hot_vector", column)

    def build_model(self):
        return OneHotEncoder(sparse=False)

    def fit(self, data):
        self.model.fit(data)
        categories = len(self.model.categories_[0])
        self.output_info = [(categories, 'softmax')]
        self.output_dimensions = categories
        self.components = categories

    def transform(self, data):
        return self.model.transform(data)

    def inverse_transform(self, data, sigma=None):
        return self.model.inverse_transform(data)


def build_embedding(name, column):
    embedding_class = _EMBEDDING_REGISTRY.get(name.lower())
    if embedding_class is None:
        raise ValueError("No Embedding model associated with the name: {}".format(name))
    else:
        return embedding_class(column)
