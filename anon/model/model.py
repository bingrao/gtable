import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
from anon.model.gan import GANModel


class AnonModel:
    def __init__(self):
        self.model = GANModel()
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = self.model.checkpoint

        self.EPOCHS = 50
        self.noise_dim = 100
        self.num_examples_to_generate = 16

        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256

    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def train(self, dataset, epochs):
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
        self.generate_and_save_images(self.model.generator, self.epochs, self.seed)

    def load_dataset(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        return train_dataset

    def run(self):
        train_dataset = self.load_dataset()
        self.train(train_dataset, self.EPOCHS)


if __name__ == "__main__":
    model = AnonModel()
    model.run()