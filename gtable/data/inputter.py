import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import pickle
from gtable.utils.constants import CATEGORICAL, ORDINAL, NUMERICAL


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def get_metadata(data, metadata):
    meta = []

    df = pd.DataFrame(data)
    for index, item in enumerate(metadata['columns']):
        column = df[index]
        if item['type'] == CATEGORICAL:

            mapper = item['i2s'] if 'i2s' in item else column.value_counts().index.tolist()
            meta.append({
                "name": item['name'],
                "index": index,
                "type": CATEGORICAL,
                "size": len(mapper),
                "i2s": mapper
            })
        elif item['type'] == ORDINAL:
            if "i2s" in item:
                mapper = item['i2s']
            else:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
            meta.append({
                "name": item['name'],
                "index": index,
                "type": ORDINAL,
                "size": len(mapper),
                "i2s": mapper
            })
        else:
            meta.append({
                "name": item['name'],
                "index": index,
                "type": NUMERICAL,
                "min": column.min(),
                "max": column.max(),
            })

    return {"columns": meta}


def category_to_number(df, cat_names=[]):
    metadata = {}
    df_num = df.copy()

    metadata['table_colums_name'] = {'y': [], 'label': df_num.columns}

    # TRANSFORM TO SET TO PREVENT DOUBLE FACTORIZATION
    for z in set(df_num.select_dtypes(include=['object']).columns.tolist() + cat_names):
        y, label = pd.factorize(df[z])
        metadata[z] = {'y': y, 'label': label}
        df_num[z] = y
    return df_num, metadata


def read_csv(csv_filename, sep=',', meta_filename=None, header=True, discrete=None):

    data = pd.read_csv(csv_filename, sep=sep, header='infer' if header else None, low_memory=False)

    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)

        discrete_columns = [
            column['name']
            for column in metadata['columns']
            if column['type'] != 'continuous'
        ]

    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]

    else:
        discrete_columns = list(data.select_dtypes(include=['object']).columns.tolist())

    return data, discrete_columns


def read_tsv(data_filename, meta_filename):
    with open(meta_filename) as f:
        column_info = f.readlines()

    column_info_raw = [
        x.replace("{", " ").replace("}", " ").split()
        for x in column_info
    ]

    discrete = []
    continuous = []
    column_info = []

    for idx, item in enumerate(column_info_raw):
        if item[0] == 'C':
            continuous.append(idx)
            column_info.append((float(item[1]), float(item[2])))
        else:
            assert item[0] == 'D'
            discrete.append(idx)
            column_info.append(item[1:])

    meta = {
        "continuous_columns": continuous,
        "discrete_columns": discrete,
        "column_info": column_info
    }

    with open(data_filename) as f:
        lines = f.readlines()

    data = []
    for row in lines:
        row_raw = row.split()
        row = []
        for idx, col in enumerate(row_raw):
            if idx in continuous:
                row.append(col)
            else:
                assert idx in discrete
                row.append(column_info[idx].index(col))

        data.append(row)

    return np.asarray(data, dtype='float32'), meta['discrete_columns']


def write_tsv(data, meta, output_filename):
    with open(output_filename, "w") as f:
        for row in data:
            for idx, col in enumerate(row):
                if idx in meta['continuous_columns']:
                    print(col, end=' ', file=f)
                else:
                    assert idx in meta['discrete_columns']
                    print(meta['column_info'][idx][int(col)], end=' ', file=f)

            print(file=f)


def pickle_save(ctx, dataset, path):
    ctx.logger.info(f'Saving dataset file: {path}')
    pickle.dump(dataset, open(f'{path}', 'wb'))


def pickle_load(ctx, path):
    ctx.logger.info(f'Loading dataset file: {path}')
    return pickle.load(open(path, 'rb'))


def load_data(path_real: str,
              path_fake: str,
              real_sep: str = ',',
              fake_sep: str = ',',
              drop_columns: List = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a real and synthetic data csv. This function makes sure that the loaded data has the same columns
    with the same data types.

    :param path_real: string path to csv with real data
    :param path_fake: string path to csv with real data
    :param real_sep: separator of the real csv
    :param fake_sep: separator of the fake csv
    :param drop_columns: names of columns to drop.
    :return: Tuple with DataFrame containing the real data and DataFrame containing the synthetic data.
    """
    # real = pd.read_csv(path_real, sep=real_sep, low_memory=False)
    # fake = pd.read_csv(path_fake, sep=fake_sep, low_memory=False)
    real, _ = read_csv(path_real, real_sep)
    fake, _ = read_csv(path_fake, fake_sep)
    if set(fake.columns.tolist()).issubset(set(real.columns.tolist())):
        real = real[fake.columns]
    elif drop_columns is not None:
        real = real.drop(drop_columns, axis=1)
        try:
            fake = fake.drop(drop_columns, axis=1)
        except:
            print(f'Some of {drop_columns} were not found on fake.index.')
        assert len(fake.columns.tolist()) == len(real.columns.tolist()), \
            f'Real and fake do not have same nr of columns: {len(fake.columns)} and {len(real.columns)}'
        fake.columns = real.columns
    else:
        fake.columns = real.columns

    for col in fake.columns:
        fake[col] = fake[col].astype(real[col].dtype)
    return real, fake
