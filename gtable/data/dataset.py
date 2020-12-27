from gtable.data.inputter import read_csv
from gtable.utils.misc import ClassRegistry
from gtable.data.transformer import build_transformer
from gtable.utils.constants import CATEGORICAL, ORDINAL
from gtable.data.inputter import get_metadata, _get_columns, pickle_load, pickle_save
import numpy as np
import pandas as pd
import urllib
import json
import abc
import os

BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        url = BASE_URL + filename

        urllib.request.urlretrieve(url, local_path)

    return loader(local_path)


def load_dataset(name, benchmark=False):
    # LOGGER.info('Loading dataset %s', name)
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    if benchmark:
        return train, data['test'], meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns


class Dataset(abc.ABC):
    def __init__(self, ctx, name):
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self._name = name

        self.train_dataset = None  # Numpy format original dataset
        self.train_metadata = None
        self.num_train_dataset = 0

        self.test_dataset = None  # Numpy format original dataset
        self.test_metadata = None
        self.num_test_dataset = 0

        self.transformer = None

        self.random_seed = self.config.seed
        self.unique_thresh = self.config.unique_thresh

        # TODO preparing training datasets in future
        self.target_col = self.config.target_col
        self.features_col = self.config.features_col

    @property
    def name(self):
        return self._name

    def load(self, inputPath: str):
        """
        Loading dataset from file system and convert it a DataFrme
        :return:
        """
        raise NotImplementedError

    def save(self, data, metadata, outputPath):
        pickle_save(self.context, {"data": data, "metadata": metadata}, outputPath)

    def preprocess(self):
        raise NotImplementedError

    def __call__(self, inputPath: str, metadata):
        # loading dataset as DataFrame

        self.metadata = metadata

        self.train_dataset, self.test_dataset = self.load(inputPath)
        self.num_train_dataset = len(self.train_dataset)
        self.num_test_dataset = len(self.test_dataset)

        self.train_metadata = get_metadata(self.train_dataset, self.metadata)

        self.test_metadata = get_metadata(self.test_dataset, self.metadata)

        self.transformer = build_transformer(self.context, self.metadata)

        self.name_columns = list(map(lambda item: item['name'], metadata['columns']))
        self.num_columns = len(self.name_columns)



        # Data clean
        self.preprocess()

        return self


_DATASET_REGISTRY = ClassRegistry(base_class=Dataset)
register_dataset = _DATASET_REGISTRY.register  # pylint: disable=invalid-name


@register_dataset(name="csv")
class CSVDataset(Dataset):
    def __init__(self, ctx, name):
        super(CSVDataset, self).__init__(ctx, name)

    def load(self, inputPath: str) -> pd.DataFrame:
        """
        Loading dataset from file system and convert it a DataFrme
        :return:
        """
        data, cat_cols = read_csv(csv_filename=inputPath, discrete=self.config.cat_cols)

        if cat_cols is None:
            self.numerical_cols = [column for column in data._get_numeric_data().columns
                                   if len(data[column].unique()) > self.unique_thresh]

            self.categorial_cols = [column for column in data.columns
                                    if column not in self.numerical_cols]
        else:
            self.categorial_cols = cat_cols
            self.numerical_cols = [column for column in data.columns
                                   if column not in self.categorial_cols]

        self.num_samples = min(self.config.num_samples, len(data)) \
            if self.config.num_samples is not None else len(data)

        self.dataset = data.sample(self.num_samples)
        self.name_columns = self.dataset.columns
        self.num_columns = len(self.name_columns)

        self.preprocess()

        return self.dataset

    def preprocess(self):
        assert self.dataset is not None
        # fill N/A with "NAN" for categorial columns
        self.dataset.loc[:, self.categorial_cols] = \
            self.dataset.loc[:, self.categorial_cols].fillna('[NAN]')

        # fill N/A with mean values for numerical columns
        self.dataset.loc[:, self.numerical_cols] = \
            self.dataset.loc[:, self.numerical_cols] \
                .fillna(self.dataset[self.numerical_cols].mean())

        # Convert categorial colums to ordinal repsentations.
        self.metadata['colums_name'] = {'y': [], 'label': self.dataset.columns}

        # TRANSFORM TO SET TO PREVENT DOUBLE FACTORIZATION
        for z in set(self.categorial_cols):
            y, label = pd.factorize(self.dataset[z])
            self.metadata[z] = {'y': y, 'label': label}
            self.dataset[z] = y


@register_dataset(name="tsv")
class TSVDataset(Dataset):
    def __init__(self, ctx, name):
        super(TSVDataset, self).__init__(ctx, name)

    def load(self, inputPath: str) -> pd.DataFrame:
        pass

    def preprocess(self):
        pass


@register_dataset(name="numpy")
class NumpyDataset(Dataset):
    def __init__(self, ctx, name):
        super(NumpyDataset, self).__init__(ctx, name)

    def load(self, inputPath: str):
        dataset = np.load(inputPath)
        return dataset['train'], dataset['test']

    def preprocess(self):
        pass


@register_dataset(name="json")
class JsonDataset(Dataset):
    def __init__(self, ctx, name):
        super(JsonDataset, self).__init__(ctx, name)

    def load(self, inputPath: str) -> pd.DataFrame:
        pass

    def preprocess(self):
        pass


@register_dataset(name="pickles")
class PickleDataset(Dataset):
    def __init__(self, ctx, name):
        super(PickleDataset, self).__init__(ctx, name)

    def load(self, inputPath: str) -> pd.DataFrame:
        pass

    def preprocess(self):
        pass


@register_dataset(name="text")
class TextDataset(Dataset):
    def __init__(self, ctx, name):
        super(TextDataset, self).__init__(ctx, name)

    def load(self, inputPath: str) -> pd.DataFrame:
        pass

    def preprocess(self):
        pass


def get_data_loader(ctx):
    name = ctx.config.data_type
    data_class = _DATASET_REGISTRY.get(name.lower())
    if data_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return data_class(ctx, name)
