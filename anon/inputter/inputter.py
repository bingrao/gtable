import glob
import os
import codecs
import math
import pandas as pd
from collections import Counter, defaultdict
from itertools import chain, cycle
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import pickle

def padding_duplicating(data, row_size):
    arr_data = np.array(data.values.tolist())

    col_num = arr_data.shape[1]

    npad = ((0, 0), (0, row_size - col_num))

    # PAdding with zero
    arr_data = np.pad(arr_data, pad_width=npad, mode='constant', constant_values=0.)

    # Duplicating Values
    for i in range(1, arr_data.shape[1] // col_num):
        arr_data[:, col_num * i: col_num * (i + 1)] = arr_data[:, 0: col_num]

    return arr_data


def reshape(data, dim):
    return data.reshape(data.shape[0], dim, -1, 1).astype('float32')


def build_dataset_iter(ctx, corpus_type, opt, is_train=True):
    """
        This returns user-defined train/validate data iterator for the trainer
        to iterate over. We implement simple ordered iterator strategy here,
        but more sophisticated strategy like curriculum learning is ok too.
        """
    # dataset_glob = opt.data + '.' + corpus_type + '.*.pkl'
    # dataset_paths = list(sorted(
    #     glob.glob(dataset_glob),
    #     key=lambda p: int(p.split(".")[-2])))
    #
    # if not dataset_paths:
    #     if is_train:
    #         raise ValueError('Training data %s not found' % dataset_glob)
    #     else:
    #         return None

    train_src = opt.data + f'.{corpus_type}' + '.src.pkl'
    train_tgt = opt.data + f'.{corpus_type}' + '.tgt.pkl'

    batch_size = opt.batch_size if is_train else opt.valid_batch_size

    dim = 28
    src_df = pickle_load(ctx, train_src)  # (nums_records, nums_attrs)
    nums_records, attrib_num = src_df.shape

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    # Normalizing Initial Data
    src_df = pd.DataFrame(min_max_scaler.fit_transform(src_df))  # (nums_records, nums_attrs)

    padded_src_df = padding_duplicating(src_df, dim * dim)   # (nums_records, dim * dim)

    features = reshape(padded_src_df, dim)   # (nums_records, dim, dim)

    labels = pickle_load(ctx, train_tgt).values

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    train_dataset = dataset.shuffle(nums_records).batch(batch_size)

    return train_dataset


def pickle_save(ctx, dataset, path):
    ctx.logger.info(f'Saving dataset file: {path}')
    pickle.dump(dataset, open(f'{path}', 'wb'))


def pickle_load(ctx, path):
    ctx.logger.info(f'Loading dataset file: {path}')
    return pickle.load(open(path, 'rb'))
