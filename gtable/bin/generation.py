#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 Bingbing Rao. All Rights Reserved.
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
from gtable.utils.results import make_leaderboard
import os


def _get_parser():
    parser = ArgumentParser(run_type="generation", description='generation.py')
    opts.config_opts(parser)
    opts.train_opts(parser)
    opts.model_opts(parser)
    opts.optimizer_opts(parser)
    opts.checkpoint_opts(parser)
    # opts.generation_opts(parser)
    opts.runtime_opts(parser)
    opts.evaluate_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    ctx = Context(opt)

    scores = []
    for i in range(opt.iterations):
        scores.append(Runner(ctx).run(i))

    lb = make_leaderboard(scores, output_path=os.path.join(ctx.output,
                                                           f"{ctx.real_name}-"
                                                           f"{ctx.app.lower()}-"
                                                           f"{ctx.config_file_name}-"
                                                           f"leaderboard.csv"))
    ctx.logger.info(f"The average evaluate metrics: \n"
                    f"{lb['accuracy_real', 'accuracy_fake', 'f1_score_real', 'f1_score_fake']}")


if __name__ == "__main__":
    main()
