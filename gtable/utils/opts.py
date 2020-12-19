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

""" Implementation of all available options """
from __future__ import print_function
import configargparse


def config_opts(parser):
    group = parser.add_argument_group('General')
    group.add('--config', '-config', required=False,
              is_config_file_arg=True, help='config file path')

    group.add('--app', '-app', type=str, choices=["CTGAN", "TVAE", "GTABLE"], default="CTGAN",
              help="The type of application to generate synthetic data")

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

    group.add('--device', type=str, default='gpu', choices=['gpu', 'cpu'],
              help="Set up device to run the application")

    group.add('--seed', '-seed', type=int, default=1337,
              help='The seed for random function')

    dataset_opts(parser)


def dataset_opts(parser):
    group = parser.add_argument_group('Dataset')

    group.add('--real_data', '-real_data', required=False, default=None,
              help='The path of real datasets')

    group.add('--fake_data', '-fake_data', type=str, required=False, default=None,
              help="The path of fake dataset")

    group.add('--attrib_num', '-attrib_num', type=int, default=0,
              help="The number of columns in the dataset. Used if the Classifer NN is active.")

    # group.add('--data', '-data', required=False, default=None,
    #           help='The input path prefix to the ".train.pkl". '
    #                'If there is no data prefix provided, '
    #                'system use real data path directly')

    group.add('--save_data', '-save_data', type=str, required=False, default=None,
              help="Output file for the prepared data")

    parser.add_argument('-t', '--tsv', action='store_true',
                        help='Load data in TSV format instead of CSV')

    parser.add_argument('--no-header', dest='header', action='store_false',
                        help='The CSV file has no header. Discrete columns will be indices.')

    group.add('--data_type', '-data_type', default="csv",
              choices=["text", "image", "csv", "vec", "code"],
              help="Type of the source input")

    group.add('--sep', '-sep', type=str, default=',',
              help="The seperator symobl between two columns")

    group.add('--drop', '-drop', type=str, nargs='+', default=None,
              help="A list of columns needed to be dropped")

    group.add('--cat_cols', '-cat_cols', type=str, nargs='+', default=None,
              help="The categorial columns")

    group.add('--features_col', '-features_col', type=str, nargs='+', default=None,
              help="The features of input for classification or regression tasks")

    group.add('--unique_thresh', '-unique_thresh', type=int, default=0,
              help='Threshold for automatic evaluation if column is numeric')

    group.add('--target_col', '-target_col', type=str, default=None,
              help="The y label for classification or regression tasks")

    group.add('--num_samples', '-num_samples', type=int, default=None,
              help='Number of samples to evaluate or generate. If none, '
                   'it will take the minimal length of both datasets and cut '
                   'the larger one off to make sure they are the same length.')

    group.add('--output', '-output', type=str, required=False,
              help="The output of generation model")

    group.add('--metadata', '-metadata', type=str, default=None,
              help="The path for metadata of the input file")

    group.add('--batch_size', '-batch_size', type=int, default=500,
              help='Maximum batch size for data generation')

    group.add("--separated_embedding", "-separated_embedding", default=False, action="store_true",
              help="Enable to use seperated embedding way for each attributes")

    group.add('--continuous_embeddding', '-continuous_embeddding', type=str,
              choices=['Bayesian_Gaussian_Norm', 'Standard_Norm'], default='Standard_Norm',
              help="The way to normalize the continuous dataset")

    group.add('--discrete_embeddding', '-discrete_embeddding', type=str,
              choices=['One_Hot_Vector'], default='One_Hot_Vector',
              help="The way to normalize the discrete dataset")

    parser.add_argument('-d', '--discrete',
                        help='Comma separated list of discrete columns, no whitespaces')

    group.add('--embedding_combine', '-embedding_combine', type=str, choices=['matrix', 'vector'],
              default='matrix', help="The way to combine columns' embedding")


def model_opts(parser):
    group = parser.add_argument_group('Model')

    group.add('--feature_size', '-feature_size', type=int, default=266,
              help="Size of last FC layer to calculate the Hinge Loss fucntion.")

    group.add("--noise_dim", "-noise_dim", type=int, default=100,
              help="The latent noise dimention")

    group.add("--input_height", "-input_height", type=int, default=7,
              help="The size of image to use (will be center cropped). [108]")

    group.add("--input_width", "-input_width", type=int, default=7,
              help="The size of image to use (will be center cropped). "
                   "If None, same value as input_height [None]")

    group.add("--output_height", "-output_height", type=int, default=7,
              help="The size of the output images to produce [64]")

    group.add("--output_width", "-output_width", type=int, default=7,
              help="The size of the output images to produce. "
                   "If None, same value as output_height [None]")

    group = parser.add_argument_group('Info Loss')

    group.add("--info_loss", "-info_loss", default=True, action="store_true",
              help="Append Info loss to generator model")

    group.add('--alpha', '-alpha', type=float, default=1.0,
              help="The weight of original GAN part of loss function [0-1.0]")

    group.add('--beta', '-beta', type=float, default=1.0,
              help="The weight of information loss part of loss function [0-1.0]")

    group.add('--delta_m', '-delta_m', type=float, default=0.0,
              help="dddd")

    group.add('--delta_v', '-delta_v', type=float, default=0.0,
              help="ddd")

    group.add('--mac', '-mac', type=float, default=0.99,
              help="Moving Average Contributions")

    group.add('--g_penalty', '-g_penalty', type=float, default=0.0,
              help='Gradient penalty weight.')

    group.add('--n_critic', '-n_critic', type=int, default=1,
              help='Critic updates per generator update.')


def optimizer_opts(parser):
    # learning rate
    group = parser.add_argument_group('Optimization')
    group.add('--learning_rate', '-learning_rate', type=float, default=1e-4,
              help="Starting learning rate. "
                   "Recommended settings: sgd = 1, adagrad = 0.1, "
                   "adadelta = 1, adam = 0.001")

    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=1e-4,
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

    group.add('--cuda_visible_devices', '-cuda_visible_devices', type=int, nargs='*',
              default=[0, 1, 2, 3],
              help="Specify the GPU visible Device ID for training task")


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('Traning')

    group.add('--train', '-train', type=str, default='VGAN',
              choices=['VGAN', 'WGAN-GP'], help="The way to train GAN model")

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

    group.add("--num_gpus", "-num_gpus", type=int, default=1,
              help="Number of GPUs to use for in-graph replication.")

    group.add("--gpu_allow_growth", "-gpu_allow_growth", default=False, action="store_true",
              help="Allocate GPU memory dynamically.")

    group.add('--epoch', '-epoch', type=int, default=10,
              help='Deprecated epochs see train_steps')

    group.add("--eval_steps", "-eval_steps", type=int, default=1000,
              help="Eval model every # steps")

    group.add("--with_eval", "-with_eval", default=False,
              action="store_true", help="Enable automatic evaluation.")

    group.add("--with_generation", "-with_generation", default=False,
              action="store_true", help="Enable generating datasets")

    group.add("--save_summary_steps", "-save_summary_steps", type=int, default=1000,
              help="Print out model accuracy informaiton by every # steps")

    group.add_argument("--sample_condition_column", default=None, type=str,
                       help="Select a discrete column name.")
    group.add_argument("--sample_condition_column_value", default=None, type=str,
                       help="Specify the value of the selected discrete column.")


def generation_opts(parser):
    group = parser.add_argument_group('Generation')


def evaluate_opts(parser):
    group = parser.add_argument_group('Evaluation')

    group.add('--classify_tasks', '-classify_tasks', type=str, nargs='+', default=None,
              choices=['logistic_regression', 'random_forest', 'decision_tree', 'mlp', 'xgboost'],
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
              choices=['mean_std', 'cumsums', 'distributions', 'correlation', 'pca'],
              help="The visual measures to show statistics of real and fake datasets")

    group.add('--continuous_statistics', '-continuous_statistics', type=str, nargs='+',
              choices=['min', 'max', 'mean', 'median', 'std', 'var'], default=None,
              help="The statistics infomation of each numerical attributes")

    group.add('--discrete_statistics', '-discrete_statistics', type=str, nargs='+',
              choices=['min', 'max', 'mean', 'median', 'std', 'var'], default=None,
              help="The statistics infomation of each categorial attributes")


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
