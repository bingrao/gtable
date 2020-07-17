from src.anonymity.anonymity import Anonymity
from src.utils.context import Context


class TCloseness(Anonymity):
    def __init__(self, ctx):
        super(TCloseness, self).__init__(ctx)

    @staticmethod
    def t_closeness(df, partition, column, global_freqs):
        total_count = float(len(partition))
        d_max = None
        group_counts = df.loc[partition].groupby(column)[column].agg('count')
        for value, count in group_counts.to_dict().items():
            p = count / total_count
            d = abs(p - global_freqs[value])
            if d_max is None or d > d_max:
                d_max = d
        return d_max

    def is_t_close(self, df, partition, sensitive_column, global_freqs, p=0.2):
        """
        :param               df: The dataframe for which to check l-diversity
        :param        partition: The partition of the dataframe on which to check l-diversity
        :param sensitive_column: The name of the sensitive column
        :param     global_freqs: The global frequencies of the sensitive attribute values
        :param                p: The maximum allowed Kolmogorov-Smirnov distance
        """
        if not sensitive_column in self.categorical:
            raise ValueError("this method only works for categorical values")
        return self.t_closeness(df, partition, sensitive_column, global_freqs) <= p

    def run(self):
        self.preprocess()
        feature_columns = ['Age', 'EmployeeNumber']
        column_x, column_y = feature_columns[:2]
        sensitive_column = 'MaritalStatus'
        full_spans = self.get_spans(self.data_df, self.data_df.index)
        global_freqs = {}
        total_count = float(len(self.data_df))
        group_counts = self.data_df.groupby(sensitive_column)[sensitive_column].agg('count')
        for value, count in group_counts.to_dict().items():
            p = count / total_count
            global_freqs[value] = p
        # Let's apply this to our dataset
        finished_t_close_partitions = self.partition_dataset(self.data_df, feature_columns, sensitive_column,
                                                             full_spans,
                                                             lambda *args: self.is_k_anonymous(
                                                                 *args) and self.is_t_close(*args,
                                                                                            global_freqs))

        dft = self.build_anonymized_dataset(self.data_df, finished_t_close_partitions, feature_columns, sensitive_column)
        dft.to_csv('../../data/t-closeness.csv')
        # Let's see how t-closeness fares
        print(dft.sort_values([column_x, column_y, sensitive_column]))


if __name__ == "__main__":
    ctx = Context("privacy-messure")
    t = TCloseness(ctx)
    t.run()
