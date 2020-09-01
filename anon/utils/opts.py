""" Implementation of all available options """
from __future__ import print_function
import sys
import configargparse


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')

    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')

    parser.add('--work_model', '-work_model', type=str, default='preprocess', required=False,
               choices=['preprocess', 'train', 'predict'],
               help="Indicator that what kinds of task is working on right now")

    parser.add('--log_file', '-log_file', type=str, default="",
               help="Output logs to a file under this path.")

    parser.add('--log_file_level', '-log_file_level', type=str,
               action=StoreLoggingLevelAction,
               choices=StoreLoggingLevelAction.CHOICES,
               default="0")
    parser.add('--project_dir', '-project_dir', type=str, default="",
               help="Root Path of the application")


def model_opts(parser):
   pass


def preprocess_opts(parser):
    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. "
                   "Options are [text|img|csv|vec|code].")
    group.add('--sep', '-sep', type=str, default=',')
    group.add('--drop', '-drop', type=list, default=None)
    group.add('--cat_names', '-cat_names', type=list, default=None)

    group.add('--train_input', '-train_input', type=str, required=True, default=None,
              help="Path(s) to the training source data")

    group.add('--valid_input', '-valid_input', type=str, required=False, default=None,
              help="Path to the validation source data")

    group.add('-target', '--target', type=str, default=None,
              help='Ticket <--> OpCarrierGroup; Civilian <--> suicide')

    group.add('--save_data', '-save_data', type=str, required=False, default=None,
              help="Output file for the prepared data")




def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')

    group.add('--data', '-data', required=True,
              help='Path prefix to the ".train.src.pkl" and '
                   '".valid.src.pkl" file path from preprocess.py')

    group.add('--train_src', '-train_src', type=str, required=False, default=None,
              help="Path(s) to the training source data")
    group.add('--train_tgt', '-train_tgt', type=str, required=False, default=None,
              help="Path(s) to the training target data")

    group.add('--valid_src', '-valid_src', type=str, required=False, default=None,
              help="Path to the validation source data")
    group.add('--valid_tgt', '-valid_tgt', type=str, required=False, default=None,
              help="Path to the validation target data")

    group.add('--epoch', '-epoch', type=int, default=64,
              help='Deprecated epochs see train_steps')





    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add('--learning_rate', '-learning_rate', type=float, default=1.0,
              help="Starting learning rate. "
                   "Recommended settings: sgd = 1, adagrad = 0.1, "
                   "adadelta = 1, adam = 0.001")
    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=0.5,
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
    group.add('--label_smoothing', '-label_smoothing', type=float, default=0.0,
              help="Label smoothing value epsilon. "
                   "Probabilities of all non-true labels "
                   "will be smoothed by epsilon / (vocab_size - 1). "
                   "Set to zero to turn off label smoothing. "
                   "For more detailed information, see: "
                   "https://arxiv.org/abs/1512.00567")
    group.add('--average_decay', '-average_decay', type=float, default=0,
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

    group.add('--train_size', '-train_size', type=int, default=sys.maxsize, help="The size of train images [np.inf]")
    group.add('--y_dim', '-y_dim', type=int, default=2, help="Number of unique labels")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add('--batch_size', '-batch_size', type=int, default=256,
              help='Maximum batch size for training')
    group.add('--batch_type', '-batch_type', default='sents',
              choices=["sents", "tokens"],
              help="Batch grouping for batch_size. Standard "
                   "is sents. Tokens will do dynamic batching")

    group.add('--input_height', '-input_height', type=int, default=16,
              help="The size of image to use (will be center cropped). [108]")

    group.add('--input_width', '-input_width', type=int, default=None,
              help="The size of image to use (will be center cropped). If None, same value as input_height [None]")

    group.add('--output_height', '-output_height', type=int, default=16,
              help="The size of the output images to produce [64]")

    group.add('--output_width', '-output_width', type=int, default=None,
              help="The size of the output images to produce. If None, same value as output_height [None]")

    group.add('--dataset', '-dataset', type=str, default="celebA",
              help="The name of dataset [celebA, mnist, lsun]")

    group.add('--checkpoint_par_dir', '-checkpoint_par_dir', type=str, default="checkpoint",
              help="Parent Directory name to save the checkpoints [checkpoint]")

    group.add('--checkpoint_dir', '-checkpoint_dir', type=str, default="",
              help="Directory name to save the checkpoints [checkpoint]")

    group.add('--sample_dir', '-sample_dir', type=str, default="samples",
              help="Directory name to save the image samples [samples]")
    group.add('--train', '-train', type=bool, default=False,
              help="True for training, False for testing [False]")
    group.add('--crop', '-crop', type=bool, default=False,
              help="True for training, False for testing [False]")
    group.add('--generate_data', '-generate_data', type=bool, default=False,
              help="True for visualizing, False for nothing [False]")

    group.add('--alpha', '-alpha', type=float, default=0.5,
              help="The weight of original GAN part of loss function [0-1.0]")
    group.add('--beta', '-beta', type=float, default=0.5,
              help="The weight of information loss part of loss function [0-1.0]")
    group.add('--delta_m', '-delta_m', type=float, default=0.5,
              help="")
    group.add('--delta_v', '-delta_v', type=float, default=0.5,
              help="")

    group.add('--test_id', '-test_id', type=str, default="5555",
              help="The experiment settings ID.Affecting the values of alpha, beta, delta_m and delta_v.")

    group.add('--label_col', '-label_col', type=int, default=-1,
              help="The column used in the dataset as the label column (from 0). Used if the Classifer NN is active.")

    group.add('--attrib_num', '-attrib_num', type=int, default=0,
              help="The number of columns in the dataset. Used if the Classifer NN is active.")
    group.add('--feature_size', '-feature_size', type=int, default=266,
              help="Size of last FC layer to calculate the Hinge Loss fucntion.")

    group.add('--shadow_gan', '-shadow_gan', type=bool, default=False,
              help="True for loading fake data from samples directory[False]")
    group.add('--shgan_input_type', '-shgan_input_type', type=int, default=0,
              help="Input for Discrimiator of shadow_gan. 1=Fake, 2=Test, 3=Train Data")


def translate_opts(parser):
   pass


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
