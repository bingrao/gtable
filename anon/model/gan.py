from tensorflow.keras import Model
import tensorflow as tf
from anon.discriminator.discriminator import Discriminator
from anon.generator.generator import Generator


class GANModel(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256
        self.noise_dim = 100
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator_optimizer = self.generator.optimizer
        self.discriminator_optimizer = self.discriminator.optimizer
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, x):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_x = self.generator(noise, training=True)

            real_output = self.discriminator(x, training=True)
            fake_output = self.discriminator(generated_x, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    @tf.function
    def test_step(self, x):
        pass

    def build_model(self):
        pass

    def discriminator_loss(self, real_output, fake_output):
        return self.discriminator.loss(real_output, fake_output)

    def generator_loss(self, fake_output):
        return self.generator.loss(fake_output)