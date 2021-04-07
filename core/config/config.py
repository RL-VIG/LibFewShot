import os
import re

import yaml
import argparse


def get_cur_path():
    return os.path.dirname(__file__)


DEFAULT_FILE = os.path.join(get_cur_path(), 'default.yaml')


class Config(object):
    def __init__(self, config_file=None, variable_dict=None, is_resume=False):
        self.is_resume = is_resume
        self.console_dict=self._load_console_dict()
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

    def _load_console_dict(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-w', '--way_num', help='way num')
        parser.add_argument('-s', '--shot_num', help='shot num')
        parser.add_argument('-q', '--query_num', help='query num')
        parser.add_argument('-bs', '--batch_size', help='batch_size')
        parser.add_argument('-es', '--episode_size', help='episode_size')

        parser.add_argument('-data', '--data_root', help='dataset path')
        parser.add_argument('-log_name', '--log_name', help='specific log dir name if necessary')
        parser.add_argument('-image_size', help='image size')
        parser.add_argument('-aug', '--augment', action='store_true')
        parser.add_argument('-aug_times', '--augment_times', help='augment times')
        parser.add_argument('-train_episode', help='train episode num')
        parser.add_argument('-test_episode', help='test episode num')
        parser.add_argument('-epochs', help='epoch num')
        parser.add_argument('-result', '--result_root', help='result path')
        parser.add_argument('-save_interval', help='checkpoint save interval')
        parser.add_argument('-log_level', help='log level in: debug, info, warning, error, critical')
        parser.add_argument('-log_interval', help='log interval')
        parser.add_argument('-gpus', '--device_ids', help='device ids')
        parser.add_argument('-n_gpu', help='gpu num') # TODO: n_gpu should be len(gpus)?
        parser.add_argument('-seed', help='seed')
        parser.add_argument('-deterministic', action='store_true', help='deterministic or not')
        args = parser.parse_args()
        return vars(args)

    def _merge_config_dict(self):
        config_dict = dict()
        config_dict.update(self.default_dict)
        config_dict.update(self.file_dict)
        config_dict.update(self.variable_dict)
        config_dict.update(self.console_dict)

        if config_dict['test_way'] is None:
            config_dict['test_way'] = config_dict['way_num']
        if config_dict['test_shot'] is None:
            config_dict['test_shot'] = config_dict['shot_num']
        if config_dict['test_query'] is None:
            config_dict['test_query'] = config_dict['query_num']

        config_dict['resume'] = self.is_resume
        return config_dict
