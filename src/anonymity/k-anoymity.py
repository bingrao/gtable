from src.anonymity.anonymity import Anonymity
from src.utils.context import Context
from os.path import dirname, abspath, join, exists
import os

class KAnonymity(Anonymity):
    def __init__(self, ctx):
        super(KAnonymity, self).__init__(ctx)

    def run(self):
        self.preprocess()

        feature_columns = ['Age', 'EmployeeNumber']
        sensitive_column = 'MaritalStatus'
        full_spans = self.get_spans(self.data_df, self.data_df.index)
        finished_partitions = self.partition_dataset(self.data_df,
                                                     feature_columns,
                                                     sensitive_column,
                                                     full_spans,
                                                     self.is_k_anonymous)
        dfn = self.build_anonymized_dataset(self.data_df,
                                            finished_partitions,
                                            feature_columns,
                                            sensitive_column)
        dfn.to_csv('../../data/k-anoymity.csv')

        print(dfn.sort_values(feature_columns + [sensitive_column]))


if __name__ == "__main__":
    ctx = Context("privacy-messure")
    k = KAnonymity(ctx)
    k.run()
