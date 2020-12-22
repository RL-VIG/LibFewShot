import os
import re

import yaml


def get_cur_path():
    return os.path.dirname(__file__)


DEFAULT_FILE = os.path.join(get_cur_path(), 'default.yaml')


class Config(object):
    def __init__(self, config_file=None, variable_dict=None):
        self.default_dict = self._load_config_files(DEFAULT_FILE)
        self.file_dict = self._load_config_files(config_file)
        self.variable_dict = self._load_variable_dict(variable_dict)
        self.config_dict = self._merge_config_dict()

    def get_config_dict(self):
        return self.config_dict

    def _load_config_files(self, config_file):
        config_dict = dict()
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        if config_file is not None:
            with open(config_file, 'r', encoding='utf-8') as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))

        return config_dict

    def _load_variable_dict(self, variable_dict):
        config_dict = dict()
        config_dict.update(variable_dict if variable_dict is not None else {})
        return config_dict

    def _merge_config_dict(self):
        config_dict = dict()
        config_dict.update(self.default_dict)
        config_dict.update(self.file_dict)
        config_dict.update(self.variable_dict)

        return config_dict
