import tensorflow as tf
import os


class Trainer:
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

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, x):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

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

        self.generator.train_loss_metrics(gen_loss)
        self.discriminator.train_loss_metrics(disc_loss)

    @tf.function
    def test_step(self, x):
        pass

    def discriminator_loss(self, real_output, fake_output):
        return self.discriminator.loss(real_output, fake_output)

    def generator_loss(self, fake_output):
        return self.generator.loss(fake_output)

    def metrics_reset(self):
        self.generator.metrics_reset()
        self.discriminator.metrics_reset()

    @classmethod
    def from_context(cls, ctx):
        raise NotImplementedError
