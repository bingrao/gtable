import tensorflow as tf
from utils.base_model import BaseModel


class Classification(BaseModel):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def build_model():
        raise NotImplementedError

    def loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss