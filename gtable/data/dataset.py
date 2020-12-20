from gtable.data.transformer import DataTransformer
from gtable.data.inputter import read_csv
import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, ctx):
        self.context = ctx
        self.config = ctx.config

        self.logging = ctx.logger
        self.transformer = DataTransformer(ctx)
        self.dataset = None
        self.column_names = None
        self.dtype = None
        self.numerical_cols = None
        self.categorial_cols = None
        self.num_samples = 0
        self.metadata = {}
        self.random_seed = self.config.seed
        self.unique_thresh = self.config.unique_thresh

        # TODO preparing training datasets in future
        self.target_col = self.config.target_col
        # self.target_col = None
        self.features_col = self.config.features_col

        self.X = None
        self.y = None

    @property
    def get_dataset(self):
        return self.dataset

    def load_dataset(self, inputPath: str) -> pd.DataFrame:
        """
        Loading dataset from file system and convert it a DataFrme
        :return:
        """
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def transform_dataset(self):
        raise NotImplementedError

    def build_dataset(self, inputPath):
        raise NotImplementedError


class CSVDataset(Dataset):
    def __init__(self, ctx, name):
        super(CSVDataset, self).__init__(ctx)
        self.dataset_name = name

    def load_dataset(self, inputPath: str):
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

        # fill N/A with "NAN" for categorial columns
        self.dataset.loc[:, self.categorial_cols] = self.dataset.loc[:, self.categorial_cols]\
            .fillna('[NAN]')

        # fill N/A with mean values for numerical columns
        self.dataset.loc[:, self.numerical_cols] = self.dataset.loc[:, self.numerical_cols]\
            .fillna(self.dataset[self.numerical_cols].mean())

        # Convert categorial colums to ordinal repsentations.
        self.metadata['colums_name'] = {'y': [], 'label': self.dataset.columns}

        # TRANSFORM TO SET TO PREVENT DOUBLE FACTORIZATION
        for z in set(self.categorial_cols):
            y, label = pd.factorize(self.dataset[z])
            self.metadata[z] = {'y': y, 'label': label}
            self.dataset[z] = y

    def preprocess(self):
        dataset = self.dataset.copy()
        if self.target_col is not None:
            self.transformer.fit(dataset.drop([self.target_col], axis=1),
                                 self.metadata, self.categorial_cols)
        else:
            self.transformer.fit(dataset, self.metadata, self.categorial_cols)

    def postprocess(self):
        raise NotImplementedError

    def transform_dataset(self):
        if not self.transformer.is_trained:
            self.preprocess()

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

    def build_dataset(self, inputPath):
        self.load_dataset(inputPath)
        self.transform_dataset()
