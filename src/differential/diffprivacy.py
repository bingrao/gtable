from src.utils.data_process import BaseDataIO
import numpy as np


class DiffPrivacy(BaseDataIO):
    def __init__(self, ctx):
        super(DiffPrivacy, self).__init__()
        self.context = ctx,
        self.logging = ctx.logger
        self.df = self.loading_data(ctx.data)
        # some fields are categorical and will require special treatment
        self.categorical = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'EnvironmentSatisfaction',
                            'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

    def preprocess(self):
        for name in self.categorical:
            self.df[name] = self.df[name].astype('category')

    def postprocess(self):
        pass

    def sensitivity(self):
        raise NotImplementedError

    def query(self):
        raise NotImplementedError

    def query_dp(self, e=1, querynum=1000):
        raise NotImplementedError

    @staticmethod
    def __laplacian_noise(x, epsilon):
        return x + np.random.laplace(0, 1.0 / epsilon, 1)[0]

    def laplace(self, col_name):
        return self.df[col_name].apply(self.__laplacian_noise, args=(2,))