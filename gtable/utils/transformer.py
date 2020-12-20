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
import numpy as np
import pandas as pd
import copy


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, ctx):
        # self.context = ctx
        self.meta = []

        self.dataframe = False
        self.column_names = None
        self.dtypes = None
        self.discrete_columns = None
        self.is_trained = False
        self.nums_trained_record = 0
        self.separated_embedding = ctx.config.separated_embedding
        self.disccret_embedding_name = ctx.config.discrete_embeddding
        self.continuous_embedding_name = ctx.config.continuous_embeddding

    def fit(self, data, metadata, discrete_colums=tuple()):
        self.output_dimensions = 0
        self.output_info = []
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
            # print("Rounding column: " + str(i))
            fake[:, i] = np.array([self.nearest_value(real[:, i], x) for x in fake[:, i]])
        return fake

    def inverse_transform(self, fake, real=None, sigmas=None, save=False):
        if self.separated_embedding:
            start = 0
            output = []
            column_names = []
            sigma = sigmas[start] if sigmas else None
            for meta in self.meta:
                dimensions = meta.output_dimensions
                columns_data = fake[:, start:start + dimensions]
                inverted = meta.inverse_transform(columns_data, sigma)
                output.append(inverted)
                column_names.append(meta.column_name)
                start += dimensions
            output = np.column_stack(output)
        else:
            column_names = self.column_names
            output = self.meta[0].inverse_transform(fake, None)

        if real is not None:
            output = self.rounding(output, real, range(output.shape[1]))

        output = pd.DataFrame(output, columns=column_names)

        org_output = copy.copy(output)
        self.metadata.pop('colums_name')
        for attr in self.metadata.keys():
            index = self.metadata[attr]['label']
            org_output[attr] = org_output[attr].apply(lambda x: index[int(x)])

        if not self.dataframe:
            return output.values, org_output.values
        else:
            return output, org_output
