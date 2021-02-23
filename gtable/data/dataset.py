from gtable.data.inputter import read_csv
from gtable.utils.misc import ClassRegistry
from gtable.data.transformer import build_transformer
from gtable.utils.constants import CATEGORICAL, ORDINAL
from gtable.data.inputter import get_metadata, _get_columns, pickle_load, pickle_save
import numpy as np
import pandas as pd
import abc


class Dataset(abc.ABC):
    def __init__(self, ctx, name):
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self._name = name
        self.dataset_name = ""

        self.train_dataset = None  # Numpy format original dataset
        self.train_metadata = None
        self.num_train_dataset = 0

        self.test_dataset = None  # Numpy format original dataset
        self.test_metadata = None
        self.num_test_dataset = 0

        self.name_columns = None
        self.num_columns = 0

        self.transformer = None

        self.random_seed = self.config.seed
        self.unique_thresh = self.config.unique_thresh

        # TODO preparing training datasets in future
        # self.target_col = self.config.target_col
        # self.features_col = self.config.features_col

    @property
    def name(self):
        return self._name

    def load(self, _input) -> (np.ndarray, np.ndarray):
        """
        Loading dataset from file system and convert it a DataFrme
        :return:
        """
        raise NotImplementedError

    def save(self, data, metadata, outputPath):
        pickle_save(self.context, {"data": data, "metadata": metadata}, outputPath)

    def split_dataset(self, label=None):

        _label = label if label is not None else self.metadata['label']

        _train_dataset = pd.DataFrame(self.train_dataset, columns=self.name_columns)
        _test_dataset = pd.DataFrame(self.test_dataset, columns=self.name_columns)

        train_x = _train_dataset.drop([_label], axis=1).values
        train_y = _train_dataset[_label]

        test_x = _test_dataset.drop([_label], axis=1).values
        test_y = _test_dataset[_label]

        return train_x, train_y, test_x, test_y

    def preprocess(self):
        raise NotImplementedError

    def __call__(self, _input, metadata):
        # loading dataset as DataFrame

        self.metadata = metadata

        self.train_dataset, self.test_dataset = self.load(_input)
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

    def load(self, _input) -> (np.ndarray, np.ndarray):
        """
        Loading dataset from file system and convert it a DataFrme
        :return:
        """
        data, cat_cols = read_csv(csv_filename=_input, discrete=self.config.cat_cols)

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

    def load(self, inputPath):
        pass

    def preprocess(self):
        pass


@register_dataset(name="numpy")
class NumpyDataset(Dataset):
    def __init__(self, ctx, name):
        super(NumpyDataset, self).__init__(ctx, name)

    def load(self, _input) -> (np.ndarray, np.ndarray):
        dataset = np.load(_input)
        return dataset['train'], dataset['test']

    def preprocess(self):
        pass


@register_dataset(name="json")
class JsonDataset(Dataset):
    def __init__(self, ctx, name):
        super(JsonDataset, self).__init__(ctx, name)

    def load(self, _input) -> (np.ndarray, np.ndarray):
        pass

    def preprocess(self):
        pass


@register_dataset(name="pickles")
class PickleDataset(Dataset):
    def __init__(self, ctx, name):
        super(PickleDataset, self).__init__(ctx, name)

    def load(self, inputPath):
        pass

    def preprocess(self):
        pass


@register_dataset(name="text")
class TextDataset(Dataset):
    def __init__(self, ctx, name):
        super(TextDataset, self).__init__(ctx, name)

    def load(self, _input) -> (np.ndarray, np.ndarray):
        pass

    def preprocess(self):
        pass


@register_dataset(name="fake")
class FakeDataset(Dataset):
    def __init__(self, ctx, name):
        super(FakeDataset, self).__init__(ctx, name)

    def load(self, _input) -> (np.ndarray, np.ndarray):
        return _input

    def preprocess(self):
        pass


def get_data_loader(ctx, _name=None):
    name = ctx.config.data_type if _name is None else _name
    data_class = _DATASET_REGISTRY.get(name.lower())
    if data_class is None:
        raise ValueError("No scorer associated with the name: {}".format(name))
    return data_class(ctx, name)
