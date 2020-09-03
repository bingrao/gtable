from utils.inputter import generate_and_save_images
from anon.modules.discriminator import ImageDisc
from anon.modules.generator import ImageGen
from anon.app.app import App
from anon.app.train import Trainer
import tensorflow as tf
import time


class ImageGAN(App):
    def __init__(self, ctx):
        super(ImageGAN, self).__init__(ctx)

    def preprocess(self):
        """
        Data Preprocess and clean task
        :return: generator dataset for traning task
        """
        pass

    def train(self):
        """
        Traning task and save checkpoint of model for future generation task
        :return:
        """
        trainer = Trainer(self.context, ImageGen(self.context), ImageDisc(self.context))
        epochs = self.config.epoch
        dataset = self.load_dataset()

        for epoch in range(epochs):
            trainer.metrics_reset()
            start = time.time()

            for image_batch in dataset:
                trainer.train_step(image_batch)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            generate_and_save_images(trainer.generator, epoch + 1, trainer.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 1:
                trainer.checkpoint.save(file_prefix=trainer.checkpoint_prefix(str(epoch)))

            self.logging.info(f'Time for epoch {epoch + 1} is {time.time() - start} sec, '
                              f'Generator Loss: {trainer.generator.train_loss_metrics.result()}, '
                              f'Discriminator Loss: {trainer.discriminator.train_loss_metrics.result()}')

        # Generate after the final epoch
        generate_and_save_images(trainer.generator, epochs, trainer.seed)

    def validation(self):
        pass

    def generation(self):
        """
        Using trained model to generate anonmymous data
        :return:
        """
        pass

    def load_dataset(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        buffer_size = train_images.shape[0]
        train_images = train_images.reshape(buffer_size, 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
            .shuffle(buffer_size).batch(self.trainer.batch_size)

        return train_dataset

    def build_app(self):
        pass