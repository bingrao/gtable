from modules.base import BaseModel
import tensorflow as tf


class Generator(BaseModel):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def build_model():
        raise NotImplementedError

    def loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
