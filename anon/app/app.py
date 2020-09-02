

class App:
    def __init__(self, ctx):
        self.context = ctx
        self.logging = ctx.logger
        self.config = ctx.config
        self.workModel = self.config.work_model
        self.app = self.build_app()

    def preprocess(self):
        """
        Data Preprocess and clean task
        :return: generator dataset for traning task
        """
        raise NotImplementedError

    def train(self):
        """
        Traning task and save checkpoint of model for future generation task
        :return:
        """
        raise NotImplementedError

    def validation(self):
        raise NotImplementedError

    def postprocess(self):
        """
        Using trained model to generate anonmymous data
        :return:
        """
        raise NotImplementedError

    def load_dataset(self):
        raise NotImplementedError

    def run(self):
        if self.workModel == "preprocess":
            self.preprocess()

        if self.workModel == "train":
            self.train()

        if self.workModel == "generation":
            self.postprocess()

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx)

    def build_app(self):
        raise NotImplementedError
