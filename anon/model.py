from anon.app import str2app


class AnonModel:
    def __init__(self, ctx):
        self.context = ctx
        self.config = self.context.config
        self.logging = self.context.logger
        self.app = self.build_app()

    def build_app(self):
        return str2app[self.config.app].from_context(self.context)

    def run(self):
        self.app.run()
