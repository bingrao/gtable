import pandas as pd


class BaseDataIO:

    @staticmethod
    def loading_data(path):
        return pd.read_csv(path, engine='c')

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError
