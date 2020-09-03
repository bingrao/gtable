from tensorflow.keras import Model
import tensorflow as tf


class BaseModel(Model):
    def __init__(self, opt=None):
        super(Model, self).__init__()
        self.model = self.build_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate,
                                                  beta_1=opt.adam_beta1,
                                                  beta_2=opt.adam_beta2)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.train_loss_metrics = tf.keras.metrics.Mean(name='loss')
        self.train_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.valid_loss_metrics = tf.keras.metrics.Mean(name='loss')
        self.valid_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    def call(self, x):
        return self.model(x)

    @staticmethod
    def build_model():
        raise NotImplementedError

    def metrics_reset(self):
        self.train_loss_metrics.reset_states()
        self.train_accuracy_metrics.reset_states()
        self.valid_loss_metrics.reset_states()
        self.valid_accuracy_metrics.reset_states()
