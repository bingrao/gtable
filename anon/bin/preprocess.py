#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import utils.opts as opts
from anon.utils.parse import ArgumentParser
import pandas as pd
from anon.utils.context import Context
from anon.inputter.inputter import pickle_save


class DataEngine:
    def __init__(self, ctx):
        self.context = ctx
        self.config = ctx.config
        self.logger = ctx.logger

        self.data_type = self.config.data_type
        self.sep = self.config.sep
        self.drop = [] if self.config.drop is None else self.config.drop
        self.cat_names = [] if self.config.cat_names is None else self.config.cat_names

        self.train_input = self.config.train_input
        self.valid_input = self.config.valid_input
        self.target = self.config.target
        self.save_data = self.config.save_data

    def build_dataset(self, corpus_type):
        assert corpus_type == "train" or corpus_type == "valid"
        if corpus_type == "train":
            input_path = self.train_input
        else:
            input_path = self.valid_input

        tgt = self.target
        input_df = pd.read_csv(input_path, sep=self.sep)
        input_df, subs = self.category_to_number(input_df)

        if tgt is not None:
            tgt_df = input_df.pop(tgt)
            subs['table_colums_name']['label'] = subs['table_colums_name']['label'].drop([tgt])
            pickle_save(self.context, tgt_df, self.save_data + f'.{corpus_type}' + '.tgt.pkl')

        pickle_save(self.context, subs, self.save_data + f'.{corpus_type}.' + 'subs.pkl')
        pickle_save(self.context, input_df, self.save_data + f'.{corpus_type}' + '.src.pkl')

    def preprocess(self):
        if self.train_input is not None:
            self.build_dataset("train")

        if self.valid_input is not None:
            self.build_dataset("valid")

    def category_to_number(self, df):
        subs = {}
        df_num = df.copy()

        subs['table_colums_name'] = {'y': [], 'label': df_num.columns}

        # TRANSFORM TO SET TO PREVENT DOUBLE FACTORIZATION
        for z in set(df_num.select_dtypes(include=['object']).columns.tolist() + self.cat_names):
            y, label = pd.factorize(df[z])
            subs[z] = {'y': y, 'label': label}
            df_num[z] = y
        return df_num, subs

    def postprocess(self):
        pass


def preprocess(ctx):
    ArgumentParser.validate_preprocess_args(ctx.config)
    engine = DataEngine(ctx)
    engine.preprocess()


def _get_parser():
    parser = ArgumentParser(model="preprocess", description='preprocess.py')
    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ctx = Context(opt)
    preprocess(ctx)


if __name__ == "__main__":
    main()
