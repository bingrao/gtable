from src.utils.data_process import BaseDataIO
from src.utils.context import Context
import pandas as pd


class Anonymity(BaseDataIO):
    def __init__(self, ctx):
        super(Anonymity, self).__init__()
        self.context = ctx,
        self.logging = ctx.logger
        self.data_df = self.loading_data(ctx.data)
        # some fields are categorical and will require special treatment
        self.categorical = set((
            'Attrition',
            'BusinessTravel',
            'Department',
            'EducationField',
            'EnvironmentSatisfaction',
            'Gender',
            'JobRole',
            'MaritalStatus',
            'Over18',
            'OverTime'
        ))

    def preprocess(self):
        for name in self.categorical:
            self.data_df[name] = self.data_df[name].astype('category')

    def postprocess(self):
        pass

    def get_spans(self, df, partition, scale=None):
        spans = {}
        for column in df.columns:
            if column in self.categorical:
                span = len(df[column][partition].unique())
            else:
                span = df[column][partition].max() - df[column][partition].min()
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    def split(self, df, partition, column):
        dfp = df[column][partition]
        if column in self.categorical:
            values = dfp.unique()
            lv = set(values[:len(values) // 2])
            rv = set(values[len(values) // 2:])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return dfl, dfr

    @staticmethod
    def is_k_anonymous(df, partition, sensitive_column, k=10):
        if len(partition) < k:
            return False
        return True

    def partition_dataset(self, df, feature_columns, sensitive_column, scale, is_valid):
        finished_partitions = []
        partitions = [df.index]
        while partitions:
            partition = partitions.pop(0)
            spans = self.get_spans(df[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = self.split(df, partition, column)
                if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    @staticmethod
    def agg_categorical_column(series):
        return [','.join(set(series))]

    @staticmethod
    def agg_numerical_column(series):
        return [series.mean()]

    def build_anonymized_dataset(self, df, partitions, feature_columns, sensitive_column, max_partitions=None):
        aggregations = {}
        for column in feature_columns:
            if column in self.categorical:
                aggregations[column] = self.agg_categorical_column
            else:
                aggregations[column] = self.agg_numerical_column
        rows = []
        for i, partition in enumerate(partitions):
            if i % 100 == 1:
                self.logging.info(f"Finished {i} partitions...")
            if max_partitions is not None and i > max_partitions:
                break
            grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
            sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column: 'count'})
            values = grouped_columns.iloc[0].to_dict()
            for sensitive_value, count in sensitive_counts[sensitive_column].items():
                if count == 0:
                    continue
                values.update({
                    sensitive_column: sensitive_value,
                    'count': count,
                })
                rows.append(values.copy())
        return pd.DataFrame(rows)




