#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import utils.opts as opts
from anon.utils.parse import ArgumentParser


def predict(opt):
    pass


def _get_parser():
    parser = ArgumentParser(model="translate", description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    predict(opt)


if __name__ == "__main__":
    main()
