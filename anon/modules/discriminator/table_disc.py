from tensorflow.keras import layers
from anon.modules.discriminator import Discriminator
import tensorflow as tf


class TableDisc(Discriminator):
    def __init__(self, ctx):
        super(Discriminator, self).__init__()
        self.context = ctx
        self.logging = ctx.logger
        self.config = ctx.config

    @staticmethod
    def build_model():
        model = tf.keras.Sequential()

        # h0
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        # h1
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        # h3
        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
