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

import configargparse as cfargparse
from configargparse import ConfigFileParser, ConfigFileParserException
import gtable.utils.opts as opts
from gtable.utils.logging import logger
from collections import OrderedDict


class CustomizedYAMLConfigFileParser(ConfigFileParser):
    """Parses YAML config files. Depends on the PyYAML module.
    https://pypi.python.org/pypi/PyYAML
    """

    def __init__(self):
        self.run_type = "generation"

    def get_syntax_description(self):
        msg = ("The config file uses YAML syntax and must represent a YAML "
               "'mapping' (for details, see http://learn.getgrav.org/advanced/yaml).")
        return msg

    def set_run_type(self, value):
        self.run_type = value

    @staticmethod
    def _load_yaml():
        """lazy-import PyYAML so that configargparse doesn't have to dependend
        on it unless this parser is used."""
        try:
            import yaml
        except ImportError:
            raise ConfigFileParserException("Could not import yaml. "
                                            "It can be installed by running 'pip install PyYAML'")
        return yaml

    def parse(self, stream):
        """Parses the keys and values from a config file."""
        yaml = self._load_yaml()

        logger.info(f"Loading Config File from {stream} ...")

        try:
            parsed_obj = yaml.safe_load(stream)

            if self.run_type in parsed_obj:
                parsed_obj = parsed_obj[self.run_type]
            else:
                raise ValueError(f"The Configure file \"{stream.name}\" "
                                 f"does not contain [{self.run_type}] parameters")
        except Exception as e:
            raise ConfigFileParserException("Couldn't parse config file: %s" % e)

        if not isinstance(parsed_obj, dict):
            raise ConfigFileParserException("The config file doesn't appear to "
                                            "contain 'key: value' pairs (aka. a YAML mapping). "
                                            "yaml.load('%s') returned type '%s' instead of 'dict'."
                                            % (getattr(stream, 'name', 'stream'),
                                               type(parsed_obj).__name__))

        result = OrderedDict()
        for key, value in parsed_obj.items():
            if isinstance(value, list):
                result[key] = value
            else:
                result[key] = str(value)

        return result

    def serialize(self, items, default_flow_style=False):
        """Does the inverse of config parsing by taking parsed values and
        converting them back to a string representing config file contents.

        Args:
            :param default_flow_style: defines serialization format (see PyYAML docs)
            :param items:
        """

        # lazy-import so there's no dependency on yaml unless this class is used
        yaml = self._load_yaml()

        # it looks like ordering can't be preserved: http://pyyaml.org/ticket/29
        items = dict(items)
        return yaml.dump(items, default_flow_style=default_flow_style)


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(self, run_type, config_file_parser_class=CustomizedYAMLConfigFileParser,
                 formatter_class=cfargparse.ArgumentDefaultsHelpFormatter, **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)
        assert run_type in ["generation", "evaluate"], f"Unsupported Run type {run_type}"
        self._config_file_parser.set_run_type(run_type)

    @classmethod
    def defaults(cls, model, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls(model)
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def update_model_opts(cls, model_opt):
        pass

    @classmethod
    def validate_model_opts(cls, model_opt):
        pass

    @classmethod
    def ckpt_model_opts(cls, model, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = cls.defaults(model, opts.model_opts)
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

    @classmethod
    def validate_train_opts(cls, opt):
        assert opt.num_gpus == 1, "Currently we only support one GPU training"

        assert (opt.save_data is None) != (opt.real_data is None), \
            "Only one of \"config.data\" and \"config.real_data\" should be provided."

        if opt.with_generation:
            assert opt.real_data is not None and \
                   opt.output is not None, "Need provide real " \
                                           "data and output data path for data generation"

    @classmethod
    def validate_evaluate_opts(cls, opt):
        if opt.classify_tasks and opt.regression_tasks:
            raise AssertionError("Classification and Regression "
                                 "evaluator cannot be enabled at the same time")

    @classmethod
    def validate_preprocess_args(cls, opt):
        pass
