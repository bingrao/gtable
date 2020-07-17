from src.anonymity.anonymity import Anonymity
from src.utils.context import Context


class LDiversity(Anonymity):
    def __init__(self, ctx):
        super(LDiversity, self).__init__(ctx)

    @staticmethod
    def diversity(df, partition, column):
        return len(df[column][partition].unique())

    def is_l_diverse(self, df, partition, sensitive_column, l=2):
        """
        :param               df: The dataframe for which to check l-diversity
        :param        partition: The partition of the dataframe on which to check l-diversity
        :param sensitive_column: The name of the sensitive column
        :param                l: The minimum required diversity of sensitive attribute values in the partition
        """
        return self.diversity(df, partition, sensitive_column) >= l

    def run(self):
        self.preprocess()

        feature_columns = ['Age', 'EmployeeNumber']
        column_x, column_y = feature_columns[:2]
        sensitive_column = 'MaritalStatus'
        full_spans = self.get_spans(self.data_df, self.data_df.index)

        # now let's apply this method to our data and see how the result changes
        finished_l_diverse_partitions = self.partition_dataset(self.data_df,
                                                               feature_columns,
                                                               sensitive_column,
                                                               full_spans,
                                                               lambda *args: self.is_k_anonymous(*args) and self.is_l_diverse(*args))

        dfl = self.build_anonymized_dataset(self.data_df, finished_l_diverse_partitions, feature_columns, sensitive_column)
        dfl.to_csv('../../data/l-diveresity.csv')
        # Let's see how l-diversity improves the anonymity of our dataset
        print(dfl.sort_values([column_x, column_y, sensitive_column]))


if __name__ == "__main__":
    ctx = Context("privacy-messure")
    l = LDiversity(ctx)
    l.run()