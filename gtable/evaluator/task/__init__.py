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


BINARY_CLASS_MODELS = {
    'logistic_regression': {
        'solver': 'lbfgs',
        'n_jobs': 2,
        'class_weight': 'balanced',
        'max_iter': 50
    },
    "random_forest": {},
    "decision_tree": {
        'max_depth': 20,
        'class_weight': 'balanced'
    },
    "mlp": {
        'hidden_layer_sizes': (100,),
        'max_iter': 50
    },
    "adaboost": {'n_estimators': 50},
    "xgboost": {},
}

MULTICLASS_MODELS = {
    'logistic_regression': {
        'max_depth': 20,
        'class_weight': 'balanced'
    },
    "random_forest": {},
    "decision_tree": {
        'max_depth': 30,
        'class_weight': 'balanced',
    },
    "mlp": {
        'hidden_layer_sizes': (100,),
        'max_iter': 50
    },
    "adaboost": {},
    "xgboost": {},
}

REGRESSION_MODELS = {
    'random_forest_regr': {
        'n_estimators': 20,
        'max_depth': 5,
        'random_state': 42
    },
    "lasso": {
        'random_state': 42
    },
    "ridge": {
        'alpha': 1.0,
        'random_state': 42
    },
    "elastic_net": {
        'random_state': 42
    }
}

_MODELS = {
    'binary_classification': BINARY_CLASS_MODELS,
    'multiclass_classification': MULTICLASS_MODELS,
    'regression': REGRESSION_MODELS
}
