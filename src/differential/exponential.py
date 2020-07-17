from src.differential.diffprivacy import DiffPrivacy
from collections import Counter
from src.utils.context import Context
import pandas as pd
import numpy as np
from random import randint
import random
import time
import math


def prove_indistinguishable(queryres1, queryres2):
    """
    proove the indistinguishable for two query results

    Arguments:
        queryres1 {[list]} -- [query 1 result]
        queryres2 {[list]} -- [query 2 result]

    Returns:
        [float] -- [probability quotient]
    """

    prob1 = Counter(queryres1)
    for key in prob1:
        prob1[key] /= len(queryres1)
    prob2 = Counter(queryres2)
    for key in prob2:
        prob2[key] /= len(queryres2)
    res = 0
    num = 0
    for key in prob1:
        if key not in prob2:
            print('no query result {} in query 2'.format(key))
            continue
        res += prob1[key] / prob2[key]
        num += 1
    res1overres2 = res / num
    res = 0
    num = 0
    for key in prob2:
        if key not in prob1:
            print('no query result {} in query 1'.format(key))
            continue
        res += prob2[key] / prob1[key]
        num += 1
    res2overres1 = res / num
    return res1overres2, res2overres1


class Exponential(DiffPrivacy):
    def __init__(self, ctx, dumpy=False):
        super(Exponential, self).__init__(ctx)
        self.col = 'Education'
        self.s = self.sensitivity()
        if dumpy:
            self.df = self.loading_data_with_drop(ctx.data)

    def loading_data_with_drop(self, path):
        random.seed(time.clock())
        df = pd.read_csv(path, engine='c')
        drop_index = randint(0, df.shape[0])
        self.logging.info(f"The drop index is {drop_index}")
        return df.drop(df.index[drop_index])

    def sensitivity(self):
        """
        calculate the sensitivity
        as the score function is #members, the sensitivity is 1

        Returns:
            [int] -- [sensitivity]
        """
        return 1

    def __exponential(self, u, e):
        """
        return exponential probability

        Arguments:
            u {[float]} -- [probability]
            e {[float]} -- [epsilon]

        Returns:
            [float] -- [exponential probability]
        """
        return np.random.exponential(e * u / (2 * self.s))

    def query(self):
        count = Counter(self.df[self.col])
        return count.most_common(1)[0][0]

    def query_dp(self, e=1, querynum=1000):
        count = Counter(self.df[self.col])
        total = sum(count.values())
        count_freq = {k: v / total for k, v in count.items()}
        candidate = list(count_freq.keys())
        res = []
        for _ in range(querynum):
            weights = [self.__exponential(freq, e) for freq in count_freq]
            weights = [w / sum(weights) for w in weights]
            res.append(np.random.choice(candidate, p=weights))
        return res

    def calc_groundtruth(self):
        """
        calculate the groundtruth
        the most frequent education value

        Returns:
            [string] -- [most frequent education value]
        """
        return self.query()

    def calc_distortion(self, queryres):
        """
        calculate the distortion

        Arguments:
            queryres {[list]} -- [query result]

        Returns:
            [float] -- [distortion]
        """
        return 1 - Counter(queryres)[self.calc_groundtruth()] / len(queryres)


if __name__ == "__main__":
    ctx = Context("de-identity")
    logging = ctx.logger

    dp = Exponential(ctx)
    reg = dp.query_dp()
    rmse = dp.calc_distortion(reg)
    logging.info(f"V0 The RMSE error: {rmse}\n")

    dp_v1 = Exponential(ctx, dumpy=True)
    reg_v1 = dp_v1.query_dp()
    rmse_v1 = dp_v1.calc_distortion(reg_v1)
    logging.info(f"V1 The RMSE error: {rmse_v1}\n")

    dp_v2 = Exponential(ctx, dumpy=True)
    reg_v2 = dp_v2.query_dp()
    rmse_v2 = dp_v2.calc_distortion(reg_v2)
    logging.info(f"V2 The RMSE error: {rmse_v2}\n")

    logging.info(f" V0 vs V1: {prove_indistinguishable(reg, reg_v1)}")

    logging.info(f" V0 vs V2: {prove_indistinguishable(reg, reg_v2)}")

    logging.info(f" V1 vs V2: {prove_indistinguishable(reg_v1, reg_v2)}")

    logging.info(f"The basical standard {math.exp(0.05)}")