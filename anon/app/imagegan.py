import tensorflow as tf
from anon.discriminator.image_disc import ImageDisc
from anon.generator.image_gen import ImageGen
from anon.app.app import App
import time
from anon.inputter.inputter import generate_and_save_images


class ImageGAN(App):
    def __init__(self, ctx):
        super(ImageGAN, self).__init__(ctx, ImageGen(ctx), ImageDisc(ctx))

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            self.metrics_reset()
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            generate_and_save_images(self.generator, epoch + 1, self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            self.logging.info(f'Time for epoch {epoch + 1} is {time.time() - start} sec, '
                              f'Generator Loss: {self.generator.train_loss_metrics.result()}, '
                              f'Discriminator Loss: {self.discriminator.train_loss_metrics.result()}')

        # Generate after the final epoch
        generate_and_save_images(self.generator, epochs, self.seed)

    def load_dataset(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        buffer_size = train_images.shape[0]
        train_images = train_images.reshape(buffer_size, 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
            .shuffle(buffer_size).batch(self.batch_size)

        return train_dataset
