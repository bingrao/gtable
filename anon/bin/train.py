#!/usr/bin/env python
"""Train models."""
import utils.opts as opts
from anon.utils.parse import ArgumentParser
from anon.utils.context import Context
from anon.model.model import AnonModel


def train(ctx):
    ArgumentParser.validate_train_opts(ctx.config)
    ArgumentParser.update_model_opts(ctx.config)
    ArgumentParser.validate_model_opts(ctx.config)
    model = AnonModel(ctx)
    model.run()


def _get_parser():
    parser = ArgumentParser(model="train", description='train.py')
    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ctx = Context(opt)
    train(ctx)


if __name__ == "__main__":
    main()
