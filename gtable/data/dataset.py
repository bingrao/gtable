from gtable.data.inputter import read_csv
import numpy as np
import pandas as pd
from gtable.utils.misc import ClassRegistry
import abc
from gtable.data.transformer import build_transformer


class Dataset(abc.ABC):
    def __init__(self, ctx, name):
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger

        self._name = name

        self.dataset = None  # pd.DataFrame
        self.name_columns = None
        self.num_columns = 1
        self.dtype = None
        self.numerical_cols = None
        self.categorial_cols = None
        self.num_samples = 0

        self.metadata = {}
        self.random_seed = self.config.seed
        self.unique_thresh = self.config.unique_thresh

        # TODO preparing training datasets in future
        self.target_col = self.config.target_col
        self.features_col = self.config.features_col

        self.X = None  # Numpy array
        self.y = None  # Numpy array

        self.transformer = build_transformer(self.context)

    @property
    def name(self):
        return self._name

    def set_transformer(self, transformer):
        self.transformer = transformer

    @property
    def get_transformer(self):
        return self.transformer

    @property
    def get_dataset(self):
        return self.dataset

    def load(self, inputPath: str) -> pd.DataFrame:
        """
        Loading dataset from file system and convert it a DataFrme
        :return:
        """
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def fit(self):
        dataset = self.dataset.copy()
        if self.target_col is not None:
            self.transformer.fit(dataset.drop([self.target_col], axis=1),
                            self.categorial_cols, self.metadata)
        else:
            self.transformer.fit(dataset, self.categorial_cols, self.metadata)

    def transform(self):
        if not self.transformer.is_trained:
            self.fit()

        dataset = self.dataset.copy()

        # Normalizing Initial Data
        self.logging.info("Transforming each column into a numberial or one-hot vector")
        if self.target_col is not None:
            self.metadata['colums_name']['label'].drop([self.target_col])
            self.X = self.transformer.transform(dataset.drop([self.target_col], axis=1))
            self.y = np.array(dataset.pop(self.target_col))
        else:
            self.X = self.transformer.transform(dataset)
            self.y = np.ones(self.num_samples)

        return self.X, self.y

    def __call__(self, inputPath: str):
        # loading dataset as DataFrame
        self.load(inputPath)

        # Data clean
        self.preprocess()

        # Normalizing the dataset
        self.transform()
        return self.dataset


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
            self.dataset.loc[:, self.numerical_cols]\
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

    def load(self, inputPath: str) -> pd.DataFrame:
        pass

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
