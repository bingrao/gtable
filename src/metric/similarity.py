import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from src.metric.measure import Measure
from src.utils.context import Context


class Similarity(Measure):
    def __init__(self, ctx):
        super(Similarity, self).__init__(ctx)
        self.attrition = self.preprocess()
        self.full_cols = self.attrition.columns.values

    @staticmethod
    def pdistcompute(attrition, cols):
        # attrition is the dataframe
        # cols is the subset of columns
        attrition = attrition[cols]
        pair_wise = pd.Series(pdist(attrition, 'cosine'))  # finding pairwise distance between data
        count = pair_wise.groupby(
            pd.cut(pair_wise, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])).count()  # grouping based on values
        # plotting
        plt.plot(np.arange(len(count)), count)  # general plot for all users.
        return count

    def privacy_apriori_analysis(self, attrition, full_cols):
        self.logging.info("<=0.5 implies COMPLETE PRIVACY")
        self.logging.info(">0.5 implies PRIVACY VIOLATING ATTRIBUTE")
        fset_80 = []  # With value greater than 0.80
        fset_50 = []  # with value greater than 0.5 but less than 80
        fset_low = []
        for i in range(len(full_cols)):
            cols = full_cols[i]
            cols = [cols] + ['YearsWithCurrManager']
            count = self.pdistcompute(attrition, cols)
            if (full_cols[i] != 'YearsWithCurrManager') & (np.sum(count) != 0):
                # YearsWithCurrManager is used as reference and ignored for analysis
                # count = 0 implies all same values for col
                # print(full_cols[i] + str(":\t") + str(count[0]/sum(count)))
                if count[0] / sum(count) >= 0.8:
                    fset_80.append(full_cols[i])
                if (count[0] / sum(count) < 0.8) & (count[0] / sum(count) >= 0.5):
                    fset_50.append(full_cols[i])
                if count[0] / sum(count) < 0.5:
                    fset_low.append(full_cols[i])
        return fset_80, fset_50, fset_low

    def privacy_attr_apriori_2(self, attrition, fset_50, fset):
        # fset -> fset_50 or fset_low
        second_list = []
        for i in range(len(fset_50)):
            for j in range(len(fset)):
                if fset_50[i] != fset[j]:
                    cols = [fset_50[i]] + [fset[j]]
                    count = self.pdistcompute(attrition, cols)
                    # print(set(cols))
                    if count[0] / sum(count) > 0.75:
                        # print(cols, str(count[0]/sum(count)))
                        second_list.append(cols[1])
        return second_list

    def get_private_attribute(self):
        private_attr = []  # Contains all list of private attributes
        fset_80, fset_50, fset_low = \
            self.privacy_apriori_analysis(self.attrition, self.full_cols)
        private_attr = fset_80

        second_list = self.privacy_attr_apriori_2(self.attrition, fset_50, fset_50)
        private_attr.append(self.most_common(second_list))  # Contains all list of private attributes
        fset_50.remove(self.most_common(second_list))

        third_list = self.privacy_attr_apriori_2(self.attrition, fset_50, fset_low)
        private_attr.append(self.most_common(third_list))

        self.logging.info(f"Private Attributes: {private_attr}")


if __name__ == "__main__":
    ctx = Context("privacy-messure")
    similarity = Similarity(ctx)
    similarity.get_private_attribute()