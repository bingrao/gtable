#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import utils.opts as opts
from anon.utils.parse import ArgumentParser
from anon.model import AnonModel
from anon.utils.context import Context


def _get_parser():
    parser = ArgumentParser(model="generation", description='generation.py')
    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.generation_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ctx = Context(opt)
    generation = AnonModel(ctx)
    generation.run()


if __name__ == "__main__":
    main()
