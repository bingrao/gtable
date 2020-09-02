import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from anon.inputter.inputter import build_dataset_iter
from anon.app.app import App
from anon.discriminator.table_disc import TableDisc
from anon.generator.table_gen import TableGen


class TableGAN(App):
    def __init__(self, ctx):
        super(TableGAN, self).__init__(ctx, TableGen(ctx), TableDisc(ctx))

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            self.metrics_reset()
            start = time.time()

            for features, labels in dataset:
                self.train_step(features)

            self.logging.info(f'Time for epoch {epoch + 1} is {time.time() - start} sec, '
                              f'Generator Loss: {self.generator.train_loss_metrics.result()}, '
                              f'Discriminator Loss: {self.discriminator.train_loss_metrics.result()}')

    def load_dataset(self):
        return build_dataset_iter(self.context, "train", self.config, True)