#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import utils.opts as opts
from anon.utils.parse import ArgumentParser
from anon.utils.context import Context
from anon.main import AnonModel


def preprocess(ctx):
    ArgumentParser.validate_preprocess_args(ctx.config)
    engine = AnonModel(ctx)
    engine.run()


def _get_parser():
    parser = ArgumentParser(model="preprocess", description='preprocess.py')
    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ctx = Context(opt)
    preprocess(ctx)


if __name__ == "__main__":
    main()
