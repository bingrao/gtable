import numpy as np
import operator
from src.metric.measure import Measure
from src.utils.context import Context


class Uniqueness(Measure):
    def __init__(self, ctx):
        super(Uniqueness, self).__init__(ctx)
        self.attrition = self.preprocess()
        self.full_cols = self.attrition.columns.values

    @staticmethod
    def unique_feat(attrition, cols):
        attrition = attrition[cols]
        return len(np.unique(attrition))

    def get_private_attribute(self):
        ulst = {}  # dictionary containing col name and values
        for i in range(len(self.full_cols)):
            cols = self.full_cols[i]
            ulst[cols] = len(self.attrition) / self.unique_feat(self.attrition, cols)

        sorted_ulst = sorted(ulst.items(), key=operator.itemgetter(1))
        # print("Attributes, Average Group Size")
        unique_attr = []
        for k, v in sorted_ulst:
            if v < 55:
                unique_attr.append(k)
        self.logging.info(f"unique_attr {unique_attr}")

    def get_private_attribute_v2(self):

        # Uniqueness fails to look for uniqueness within an attribute
        # Third Metric - Finding uniqueness within an attribute - in imbalanced dataset
        # df = self.attrition.groupby('Age')['Age'].count()
        # print(df.min())

        alst = {}
        for i in range(len(self.full_cols)):
            cols = self.full_cols[i]
            mval = (self.attrition.groupby(cols)[cols].count()).min()
            # print(cols + str(':') + str(mval))
            alst[cols] = 1 / mval

        sorted_alst = sorted(alst.items(), key=operator.itemgetter(1))
        # print("Attributes, Average Group Size")
        imbalance_attr = []
        threshold = 0.2  # (1 in 20 records)
        for k, v in sorted_alst:
            if v > threshold:
                # print(k,v)
                imbalance_attr.append(k)
        imbalance_attr

        clst = {}
        threshold = 0.95  # (1 in 10 records)
        for i in range(len(self.full_cols)):
            for j in range(len(self.full_cols)):
                if self.full_cols[i] not in imbalance_attr:
                    if self.full_cols[j] not in imbalance_attr:
                        if self.full_cols[i] != self.full_cols[j]:
                            cols = [self.full_cols[i]] + [self.full_cols[j]]
                            mval = (self.attrition.groupby(cols)[cols].count()).min()
                            value = 1 / mval[0]
                            if value > threshold:
                                # print(str(cols) + str(value))
                                if cols[0] not in clst.keys():
                                    clst[cols[0]] = 1
                                else:
                                    clst[cols[0]] = clst[cols[0]] + 1
        sorted_clst = sorted(clst.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_clst)
        second_attr = []
        count = 0
        for k, v in sorted_clst:
            if count < 5:
                second_attr.append(k)
                count = count + 1
            else:
                break
        self.logging.info(f"second_attr = {second_attr}")


if __name__ == "__main__":
    ctx = Context("privacy-messure")
    unique = Uniqueness(ctx)
    unique.get_private_attribute()
    unique.get_private_attribute_v2()