from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf


class Classification(Model):
    def __init__(self, ctx):
        super(Model, self).__init__()
        self.context = ctx
        self.logging = ctx.logger
        self.config = ctx.config
        self.model = self.build_tablegan_model()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def call(self, x):
        return self.model(x)

    @staticmethod
    def build_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    @staticmethod
    def build_tablegan_model():
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

    @staticmethod
    def loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
