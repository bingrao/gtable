#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 Unknot.id Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import unicode_literals
import gtable.utils.opts as opts
from gtable.utils import ArgumentParser
from gtable.bin import Runner
from gtable.utils import Context


def _get_parser():
    parser = ArgumentParser(run_type="evaluate", description='evaluate.py')
    opts.config_opts(parser)
    opts.evaluate_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ArgumentParser.validate_evaluate_opts(opt)
    ctx = Context(opt)
    Runner(ctx).run()


if __name__ == "__main__":
    main()
