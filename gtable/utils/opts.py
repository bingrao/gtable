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

""" Implementation of all available options """
from __future__ import print_function
import configargparse
import torch


def config_opts(parser):
    group = parser.add_argument_group('General')
    group.add('--config', '-config', required=False,
              is_config_file_arg=True, help='config file path')

    group.add('--app', '-app', type=str, choices=["CTGAN", "TVAE", "GTABLE", "TABLEGAN"],
              default="CTGAN", help="The type of application to generate synthetic data")

    group.add('--run_type', '-run_type', type=str, default='generation', required=False,
              choices=['generation', 'evaluate'],
              help="Indicator that what kinds of task is working on right now")

    group.add('--log_file', '-log_file', type=str, default=None,
              help="Output logs to a file under this path. By deault, "
                   "the output will be redirected to {logs} folder in the working directory")

    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")

    group.add('--project_dir', '-project_dir', type=str, default="",
              help="Root path of runing application")

    group.add('--seed', '-seed', type=int, default=1337,
              help='The seed for random function')

    group.add_argument('-i', '--iterations', type=int, default=3,
                       help='Number of iterations.')

    dataset_opts(parser)


def dataset_opts(parser):
    group = parser.add_argument_group('Dataset')

    group.add('--real_data', '-real_data', required=False, default=None,
              help='The path of real datasets')

    group.add('--fake_data', '-fake_data', type=str, required=False, default=None,
              help="The path of fake dataset")

    group.add('--metadata', '-metadata', type=str, default=None,
              help="The path for metadata of the input file")

    group.add('--data_type', '-data_type', default="csv",
              choices=["csv", "tsv", "numpy", "json", "text"],
              help="Type of the source input")

    group.add('--output', '-output', type=str, required=False,
              help="The output of generation model")

    group.add('--num_samples', '-num_samples', type=int, default=None,
              help='Number of samples to evaluate or generate. If none, '
                   'it will take the minimal length of both datasets and cut '
                   'the larger one off to make sure they are the same length.')

    group.add_argument('--header', '-header', dest='header', action='store_false',
                       help='The CSV file has no header. Discrete columns will be indices.')

    group.add('--sep', '-sep', type=str, default=',',
              help="The seperator symobl between two columns")

    group.add('--drop', '-drop', type=str, nargs='+', default=None,
              help="A list of columns needed to be dropped")

    group.add('--cat_cols', '-cat_cols', type=str, nargs='+', default=None,
              help="The categorial columns")

    group.add('--features_col', '-features_col', type=str, nargs='+', default=None,
              help="The features of input for classification or regression tasks")

    group.add('--target_col', '-target_col', type=str, default=None,
              help="The y label for classification or regression tasks")

    group.add('--unique_thresh', '-unique_thresh', type=int, default=0,
              help='Threshold for automatic evaluation if column is numeric')

    group.add('--batch_size', '-batch_size', type=int, default=500,
              help='Maximum batch size for data generation')

    group = parser.add_argument_group('Dataset Transformer')

    group.add('--transformer_type', '-transformer_type', default="normal",
              choices=["general", "gmm", "tablegan", "bgm"],
              help="Type of the source input")

    group.add("--unify_embedding", "-unify_embedding", type=str, default=None,
              choices=['Bayesian_Gaussian_Norm', 'Gaussian_Norm', 'MinMax_Norm', 'One_Hot'],
              help="Enable to use a identifical and unified embedding way for "
                   "each attributes.")

    group.add('--numerical_embeddding', '-numerical_embeddding', type=str,
              default='Bayesian_Gaussian_Norm',
              choices=['Bayesian_Gaussian_Norm', 'Gaussian_Norm',
                       'MinMax_Norm', 'KBins_Discretizer', 'Power_Transformer'],
              help="The way to normalize the numerical dataset")

    group.add('--categorial_embeddding', '-categorial_embeddding', type=str,
              choices=['Ordinal', 'MinMax_Norm', 'One_Hot'], default='One_Hot',
              help="The way to normalize the categorial discrete dataset")

    group.add('--ordinal_embeddding', '-ordinal_embeddding', type=str,
              choices=['Ordinal', 'MinMax_Norm', 'One_Hot'], default='Ordinal',
              help="The way to normalize the ordinal discrete dataset")

    group.add('--embedding_combine', '-embedding_combine', type=str, choices=['matrix', 'vector'],
              default='matrix', help="The way to combine columns' embedding")

    group.add('--n_clusters', '-n_clusters', type=int, default=10,
              help='The number of clusters in GMM model')

    group.add('--epsilon', '-epsilon', type=float, default=0.005,
              help="")

    group.add("--num_channels", "-num_channels", type=int, default=64,
              help="The number of channels for tablegan")


def model_opts(parser):
    model = parser.add_argument_group('Model')

    model.add('--feature_size', '-feature_size', type=int, default=256,
              help="Size of last FC layer to calculate the Hinge Loss fucntion.")

    model.add('--noise', '-noise', default='normal', choices=['normal', 'gmm'],
              help="Generating noise input method.")

    model.add("--noise_dim", "-noise_dim", type=int, default=128,
              help="The latent noise dimention")
    ctgan = parser.add_argument_group('ctgan')
    tablegan = parser.add_argument_group('tablegan')

    gtable = parser.add_argument_group('gtable')
    gtable.add("--gen_layers", "-gen_layers", type=int, default=2,
               help="The number of generator layers")

    gtable.add("--gen_attention", "-gen_attention", default=False, action="store_true",
               help="Enable generator's Self-attention ")

    gtable.add("--gen_dim", "-gen_dim", type=int, default=256,
               help="The dimention of a generator layer")

    gtable.add("--dis_layers", "-dis_layers", type=int, default=2,
               help="The number of discriminator layers")

    gtable.add("--dis_dim", "-dis_dim", type=int, default=256,
               help="The dimention of a discriminator layer")

    gtable.add("--dis_pack", "-dis_pack", type=int, default=10,
               help="The number of packages in a discriminator layer")

    gtable.add("--dis_attention", "-dis_attention", default=False, action="store_true",
               help="Enable discriminator's Self-attention")

    gtable.add("--gtable_model", "-gtable_model", type=str, default='gtable_stardard',
               choices=['gtable_standard', 'gtable_attention', 'gtable_transformer'],
               help="Enable discriminator's Self-attention")

    gtable.add("--condition_generator", "-condition_generator", default=False, action="store_true",
               help="Enable a conditional generator for data sample")

    gtable.add('--layers_count', type=int, default=6)
    gtable.add('--d_model', type=int, default=128)
    gtable.add("--head", "-head", type=int, default=8,
               help="The number of head in multiple attention head")
    gtable.add('--d_ff', type=int, default=1024)
    gtable.add('--dropout', '-dropout', type=float, default=0.1,
               help="the ratio of Drop model")


def optimizer_opts(parser):
    # learning rate
    group = parser.add_argument_group('Optimization')

    group.add('--optim', '-optim', default='adam',
              choices=['sgd', 'adagrad', 'adadelta', 'adam'],
              help="Optimization method.")

    group.add('--learning_rate', '-learning_rate', type=float, default=1e-5,
              help="Starting learning rate. "
                   "Recommended settings: sgd = 1, adagrad = 0.1, "
                   "adadelta = 1, adam = 0.001")

    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=1e-6,
              help="If update_learning_rate, decay learning rate by "
                   "this much if steps have gone past "
                   "start_decay_steps")

    group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
              help="The beta1 parameter used by Adam. "
                   "Almost without exception a value of 0.9 is used in "
                   "the literature, seemingly giving good results, "
                   "so we would discourage changing this value from "
                   "the default without due consideration.")

    group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
              help='The beta2 parameter used by Adam. '
                   'Typically a value of 0.999 is recommended, as this is '
                   'the value suggested by the original paper describing '
                   'Adam, and is also the value adopted in other frameworks '
                   'such as Tensorflow and Keras, i.e. see: '
                   'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
                   'Optimizer or https://keras.io/optimizers/ . '
                   'Whereas recently the paper "Attention is All You Need" '
                   'suggested a value of 0.98 for beta2, this parameter may '
                   'not work well for normal models / default '
                   'baselines.')


def checkpoint_opts(parser):
    group = parser.add_argument_group('Checkpoints')

    group.add('--checkpoint_dir', '-checkpoint_dir', type=str, default="checkpoints",
              help="The output directory for the updated checkpoint.")

    group.add("--checkpoint_path", "-checkpoint_path", type=str, default=None,
              help=("Specific checkpoint or model directory to load "
                    "(when a directory is set, the latest checkpoint is used)."))

    group.add("--keep_checkpoint_max", "-keep_checkpoint_max", type=int, default=8,
              help="The maximal number of checkpoints to average.")

    group.add('--save_checkpoints_steps', '-save_checkpoints_steps', type=int, default=1000,
              help="""Save a checkpoint every X steps""")

    group.add("--average_last_checkpoints", "-average_last_checkpoints", type=int, default=8,
              help="")

    group.add("--max_count_checkpoints", "-max_count_checkpoints", type=int, default=10000,
              help="The maximal number of checkpoints to average.")

    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')


def runtime_opts(parser):
    group = parser.add_argument_group('Runtime Backend')

    group.add('--sample_dir', '-sample_dir', type=str, default="samples",
              help="Directory name to save the image samples [samples]")

    group.add_argument('--device', '-device', type=str, choices=['cuda', 'cpu'],
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    group.add('--cuda_visible_devices', '-cuda_visible_devices', type=int, nargs='*',
              default=[0, 1, 2, 3], help="Specify the GPU visible Device ID for training task")

    group.add("--num_gpus", "-num_gpus", type=int, default=1,
              help="Number of GPUs to use for in-graph replication.")

    group.add("--gpu_allow_growth", "-gpu_allow_growth", default=False, action="store_true",
              help="Allocate GPU memory dynamically.")


def train_opts(parser):
    """ Training and saving options """
    group = parser.add_argument_group('Traning')

    # group.add('--train', '-train', type=str, default='VGAN',
    #           choices=['VGAN', 'WGAN-GP'], help="The way to train GAN model")

    group.add('--label_smoothing', '-label_smoothing', type=float, default=0.0,
              help="Label smoothing value epsilon. "
                   "Probabilities of all non-true labels "
                   "will be smoothed by epsilon / (vocab_size - 1). "
                   "Set to zero to turn off label smoothing. "
                   "For more detailed information, see: "
                   "https://arxiv.org/abs/1512.00567")

    group.add('--moving_average_decay', '-moving_average_decay', type=float, default=0.9999,
              help="Moving average decay. "
                   "Set to other than 0 (e.g. 1e-4) to activate. "
                   "Similar to Marian NMT implementation: "
                   "http://www.aclweb.org/anthology/P18-4020 "
                   "For more detail on Exponential Moving Average: "
                   "https://en.wikipedia.org/wiki/Moving_average")

    group.add('--average_every', '-average_every', type=int, default=1,
              help="Step for moving average. "
                   "Default is every update, "
                   "if -average_decay is set.")

    group.add('--epochs', '-epochs', type=int, default=10,
              help='Deprecated epochs see train_steps')

    group.add("--eval_steps", "-eval_steps", type=int, default=1000,
              help="Eval model every # steps")

    group.add("--save_summary_steps", "-save_summary_steps", type=int, default=1000,
              help="Print out model accuracy informaiton by every # steps")

    group.add_argument("--sample_condition_column", default=None, type=str,
                       help="Select a discrete column name.")
    group.add_argument("--sample_condition_column_value", default=None, type=str,
                       help="Specify the value of the selected discrete column.")

    group.add('--g_penalty', '-g_penalty', type=float, default=1.0,
              help='Gradient penalty weight.')

    group.add('--n_critic', '-n_critic', type=int, default=1,
              help='Critic updates per generator update.')


# def generation_opts(parser):
#     # group = parser.add_argument_group('Generation')
#     pass


def evaluate_opts(parser):
    group = parser.add_argument_group('Evaluation')

    group.add('--classify_tasks', '-classify_tasks', type=str, nargs='+', default=None,
              choices=['logistic_regression', 'random_forest',
                       'decision_tree', 'mlp', 'xgboost', 'adaboost'],
              help="The valid classification task to evaluate the performance "
                   "of synthetic and original datasets")

    group.add('--classify_scores', '-classify_scores', type=str, nargs='+',
              choices=['ROC_AUC', 'F1_Score', 'Accuracy', 'AUC', 'Confusion_Matrix',
                       'Precision_Score', 'Recall_Score'], default='F1_Score',
              help="The calcuate the score of y and y_pred for classifciation task")

    group.add('--regression_tasks', '-regression_tasks', type=str, nargs='+', default=None,
              choices=['random_forest_regr', 'lasso', 'ridge', 'elastic_net'],
              help="The valid regression task to evaluate the performance "
                   "of synthetic and original datasets")

    group.add('--regression_scores', '-regression_scores', type=str, nargs='+',
              choices=['mse', 'mae', 'mape', 'rmse'], default='rmse',
              help="The calcuate the score of y and y_pred for regression task")

    group.add('--visual', '-visual', type=str, nargs='+', default=None,
              choices=['mean_std', 'cumsums', 'distributions', 'correlation', 'pca', 'variance'],
              help="The visual measures to show statistics of real and fake datasets")

    group.add('--numerical_statistics', '-numerical_statistics', type=str, nargs='+',
              choices=['min', 'max', 'mean', 'median', 'std', 'var'], default=None,
              help="The statistics infomation of each numerical attributes")


class StoreLoggingLevelAction(configargparse.Action):
    """ Convert string to logging level """
    import logging
    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class DeprecateAction(configargparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)
