from src.differential.diffprivacy import DiffPrivacy
from src.utils.context import Context
import pandas as pd
import numpy as np
from random import randint
import random
import time
import math


class Laplace(DiffPrivacy):
    def __init__(self, ctx, dumpy=False):
        super(Laplace, self).__init__(ctx)
        self.col = 'Age'
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
        age = self.df[self.col]
        oldest_age = max(age)
        age_25 = age.where(lambda x: x > 25).dropna()
        return oldest_age / len(age_25)

    def __laplacian_noise(self, e):
        """
        add laplacian_noise
        """
        return np.random.laplace(self.s / e)

    def query(self):
        age = self.df[self.col]
        agegt25 = age.where(lambda x: x > 25).dropna()
        avgage = sum(agegt25) / len(agegt25)
        return avgage

    def query_dp(self, e=1, nums_query=1000):
        ground_true = self.query()
        res = []
        for _ in range(nums_query):
            res.append(round(ground_true + self.__laplacian_noise(e), 2))
        return res

    def calc_groundtruth(self):
        """
        calculate the true average age above 25 without adding noise

        Returns:
            [float] -- [true average age greater than 25]
        """
        return self.query()

    def calc_distortion(self, queryres):
        """
        calcluate the distortion
        use RMSE here

        Arguments:
            queryres {[list]} -- [query result]

        Returns:
            [float] -- [rmse value]
        """

        groundtruth = self.calc_groundtruth()
        rmse = (sum((res - groundtruth) ** 2 for res in queryres) / len(queryres)) ** (1 / 2)
        return rmse


def prove_indistinguishable(queryres1, queryres2, bucketnum=20):
    """
    proove the indistinguishable for two query results

    Arguments:
        queryres1 {[list]} -- [query 1 result]
        queryres2 {[list]} -- [query 2 result]

    Keyword Arguments:
        bucketnum {int} -- [number of buckets used to calculate the probability] (default: {20})

    Returns:
        [float] -- [probability quotient]
    """

    maxval = max(max(queryres1), max(queryres2))
    minval = min(min(queryres1), min(queryres2))
    count1 = [0 for _ in range(bucketnum)]
    count2 = [0 for _ in range(bucketnum)]
    for val1, val2 in zip(queryres1, queryres2):
        count1[math.floor((val1 - minval + 1) / ((maxval - minval + 1) / bucketnum)) - 1] += 1
        count2[math.floor((val2 - minval + 1) // ((maxval - minval + 1) / bucketnum)) - 1] += 1
    prob1 = list(map(lambda x: x / len(queryres1), count1))
    prob2 = list(map(lambda x: x / len(queryres2), count2))

    res1overres2 = sum(p1 / p2 for p1, p2 in zip(prob1, prob2) if p2 != 0) / bucketnum
    res2overres1 = sum(p2 / p1 for p1, p2 in zip(prob1, prob2) if p1 != 0) / bucketnum
    return res1overres2, res2overres1


if __name__ == "__main__":
    ctx = Context("de-identity")
    logging = ctx.logger

    dp = Laplace(ctx)
    reg = dp.query_dp()
    rmse = dp.calc_distortion(reg)
    logging.info(f"V0 The RMSE error: {rmse}\n")

    dp_v1 = Laplace(ctx, dumpy=True)
    reg_v1 = dp_v1.query_dp()
    rmse_v1 = dp_v1.calc_distortion(reg_v1)
    logging.info(f"V1 The RMSE error: {rmse_v1}\n")

    dp_v2 = Laplace(ctx, dumpy=True)
    reg_v2 = dp_v2.query_dp()
    rmse_v2 = dp_v2.calc_distortion(reg_v2)
    logging.info(f"V2 The RMSE error: {rmse_v2}\n")

    logging.info(f" V0 vs V1: {prove_indistinguishable(reg, reg_v1)}")

    logging.info(f" V0 vs V2: {prove_indistinguishable(reg, reg_v2)}")

    logging.info(f" V1 vs V2: {prove_indistinguishable(reg_v1, reg_v2)}")

    logging.info(f"The basical standard {math.exp(0.05)}")
