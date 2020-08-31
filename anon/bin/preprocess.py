#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

from anon.utils.logging import init_logger
import utils.opts as opts
from anon.utils.parse import ArgumentParser


def preprocess(opt):
    ArgumentParser.validate_preprocess_args(opt)
    init_logger(opt.log_file)
    pass


def _get_parser():
    parser = ArgumentParser(model="preprocess", description='preprocess.py')
    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    preprocess(opt)


if __name__ == "__main__":
    main()
