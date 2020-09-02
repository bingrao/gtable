from utils.inputter import build_dataset_iter
from anon.modules.discriminator import TableDisc
from anon.modules.generator import TableGen
from anon.app.train.trainer import Trainer
from anon.app.preprocess import PreProcessor
from anon.app.app import App
import time


class TableGAN(App):
    def __init__(self, ctx):
        super(TableGAN, self).__init__(ctx)

    def preprocess(self):
        """
        Data Preprocess and clean task
        :return: generator dataset for traning task
        """
        preprocessor = PreProcessor(self.context)
        preprocessor.run()

    def train(self):
        """
        Traning task and save checkpoint of model for future generation task
        :return:
        """
        trainer = Trainer(self.context, TableGen(self.context), TableDisc(self.context))
        epochs = self.config.epoch
        dataset = self.load_dataset()
        for epoch in range(epochs):
            trainer.metrics_reset()
            start = time.time()

            for features, labels in dataset:
                trainer.train_step(features)

            self.logging.info(f"Time for epoch {epoch + 1} is {time.time() - start} sec, "
                              f"Generator Loss: {trainer.generator.train_loss_metrics.result()}, "
                              f"Discriminator Loss: {trainer.discriminator.train_loss_metrics.result()}")

    def validation(self):
        raise NotImplementedError

    def postprocess(self):
        """
        Using trained model to generate anonmymous data
        :return:
        """
        raise NotImplementedError

    def load_dataset(self):
        return build_dataset_iter(self.context, "train", self.config, True)

    def build_app(self):
        pass
