import tensorflow as tf
import os
import math
import numpy as np
from sklearn import preprocessing
import pickle
import pandas as pd
from anon.utils.inputter import pickle_load


class DataGenerator:
    def __init__(self, ctx, generator, discriminator):
        self.context = ctx
        self.logging = ctx.logger
        self.config = ctx.config
        self.workModel = self.config.work_model
        self.appName = self.config.app

        self.generator = generator
        self.discriminator = discriminator

        self.generator_optimizer = self.generator.optimizer
        self.discriminator_optimizer = self.discriminator.optimizer

        self.checkpoint_prefix = os.path.join(self.context.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.noise_dim = 100
        self.num_examples_to_generate = 16

        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        self.batch_size = ctx.config.batch_size
        self.epoch = ctx.config.epoch

    def restore_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.context.checkpoint_dir)).assert_consumed()

    @staticmethod
    def nearest_value(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def rounding(self, fake, real, column_list):
        # max_row = min( fake.shape[0], real.shape[0])

        for i in column_list:
            print("Rounding column: " + str(i))
            fake[:, i] = np.array([self.nearest_value(real[:, i], x) for x in fake[:, i]])

        return fake

    def generation(self):
        self.logging.info("Start Generatig Data .... ")

        origin_data_path = self.config.data + '.train.src.pkl'

        if os.path.exists(origin_data_path):
            origin_data = pickle_load(self.context, origin_data_path)
        else:
            print("Error Loading Dataset !!")
            exit(1)

        input_size, nums_attr = origin_data.shape
        dim = 28  # 8
        merged_data = np.ndarray([self.config.batch_size * (input_size // self.config.batch_size),
                                  dim, dim], dtype=float)  # 64 * 234 * 16 * 16

        save_dir = f"{self.context.project_dir}/data/{self.config.dataset}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(input_size // self.config.batch_size):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1, size=(self.config.batch_size, 100))

            zero_labeles = 0.5 # model.zero_one_ratio

            y = np.ones((self.config.batch_size, 1))

            y[: int(zero_labeles * self.config.batch_size)] = 0
            np.random.shuffle(y)

            print("y shape " + str(y.shape))
            y = y.astype('int16')

            y_one_hot = np.zeros((self.config.batch_size, 2))

            # y indicates the index of ones in y_one_hot : in this case y_dim =2 so indexe are 0 or 1
            y_one_hot[np.arange(self.config.batch_size), y] = 1

            samples = self.generator(z_sample).numpy()

            # Merging Data for each batch size
            merged_data[idx * self.config.batch_size: (idx + 1) * self.config.batch_size] = \
                samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2])  # 234 * 64 * 16 *16

        # All generated data is ready in merged_data , now reshape it to a tabular marix

        fake_data = merged_data.reshape(merged_data.shape[0], merged_data.shape[1] * merged_data.shape[2])

        # Selecting the correct number of atributes (used in training)
        fake_data = fake_data[:, : nums_attr]

        print(" Fake Data shape= " + str(fake_data.shape))

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        min_max_scaler.fit(origin_data)

        # Fake Gen --> Scaling --> Rounding --> 1) Classification , 2)-->Normalizaing --> ( Euclidian Distance, CDF)
        # transforming data back to original scale
        scaled_fake = min_max_scaler.inverse_transform(fake_data)

        # Rounding Data
        round_columns = range(scaled_fake.shape[1])

        round_scaled_fake = self.rounding(scaled_fake, origin_data.values, round_columns)

        rsf_out = pd.DataFrame(round_scaled_fake)

        subs_path = f'{self.config.project_dir}/data/{self.config.dataset}/{self.config.dataset}.train.subs.pkl'
        if os.path.exists(subs_path):
            import pickle
            subs = pickle.load(open(subs_path, "rb"))
            columns_name = subs['table_colums_name']['label']
            rsf_out.columns = columns_name
            rsf_out.to_csv(f'{save_dir}/{self.config.dataset}_{self.config.test_id}_fake.csv', index=False, sep=',')
            subs.pop('table_colums_name')

            for attr in subs.keys():
                index = subs[attr]['label']
                rsf_out[attr] = rsf_out[attr].apply(lambda x: index[int(x)])
            rsf_out.to_csv(f'{save_dir}/{self.config.dataset}_{self.config.test_id}_fake_with_label.csv', index=False, sep=',')
        else:
            print(f"There is not subs file for recover original data: {subs_path}")
            rsf_out.to_csv(f'{save_dir}/{self.config.dataset}_{self.config.test_id}_fake.csv', index=False, sep=',')

        self.logging.info("Generated Data shape = " + str(round_scaled_fake.shape))