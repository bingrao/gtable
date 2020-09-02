import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from anon.model.gan import GANModel
from anon.inputter.inputter import build_dataset_iter


class AnonModel:
    def __init__(self, ctx):

        self.config = ctx.config
        self.logging = ctx.logger
        self.context = ctx

        self.model = GANModel(ctx)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = self.model.checkpoint

        self.noise_dim = 100
        self.num_examples_to_generate = 16

        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        # fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def train(self, dataset, epochs):
        self.train_tablegan(dataset, epochs)

    def train_single(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.model.train_step(image_batch)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            self.generate_and_save_images(self.model.generator, epoch + 1, self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        self.generate_and_save_images(self.model.generator, epochs, self.seed)

    def train_tablegan(self, dataset, epochs):
        for epoch in range(epochs):
            self.model.gen_loss.reset_states()
            self.model.disc_loss.reset_states()
            start = time.time()

            for image_batch in dataset:
                self.model.train_step_tablegan(image_batch)

            self.logging.info(f'Time for epoch {epoch + 1} is {time.time() - start} sec, '
                              f'Generator Loss: {self.model.gen_loss.result()}, Discriminator Loss: {self.model.disc_loss.result()}')

    def load_dataset(self):
        return self.load_tabular_data()

    def load_minist_dataset(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        buffer_size = train_images.shape[0]
        train_images = train_images.reshape(buffer_size, 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
            .shuffle(buffer_size).batch(self.config.batch_size)

        return train_dataset

    def load_tabular_data(self):
        return build_dataset_iter(self.context, "train", self.config, True)

    def run(self):
        self.train(self.load_dataset(), self.config.epoch)
