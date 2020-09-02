from utils.inputter import pickle_save
import pandas as pd


class PreProcessor:
    def __init__(self, ctx):
        self.context = ctx
        self.logging = ctx.logger
        self.config = ctx.config
        self.workModel = self.config.work_model
        self.appName = self.config.app

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

    def run(self):
        if self.train_input is not None:
            self.build_dataset("train")

        if self.valid_input is not None:
            self.build_dataset("valid")