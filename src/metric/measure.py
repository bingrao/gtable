import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.data_process import BaseDataIO


class Measure(BaseDataIO):
    def __init__(self, ctx):
        super(Measure, self).__init__()
        self.context = ctx
        self.logging = ctx.logger
        self.project_dir = ctx.project_dir
        self.data = ctx.data

    def preprocess(self):
        # Loading data from CVS file
        attrition = self.loading_data(self.data)
        # encoding on all except Age
        attrition_encoded = attrition.iloc[:, 1:].apply(LabelEncoder().fit_transform)
        return pd.concat([attrition.iloc[:, 0], attrition_encoded], axis=1, sort=False)

    @staticmethod
    def most_common(lst):
        if len(lst) == 0:
            return 0
        else:
            return max(set(lst), key=lst.count)

    def get_private_attribute(self):
        pass

    def __str__(self):
        return f"Project Directory {self.project_dir}, data {self.data}"


