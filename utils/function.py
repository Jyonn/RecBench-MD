import os
import random
import sys

import numpy as np
import torch
from pigmento import pnt

from process.goodreads_processor import GoodreadsProcessor
from process.microlens_processor import MicroLensProcessor
from process.mind_processor import MINDProcessor
from process.movielens_processor import MovieLensProcessor
from process.steam_processor import SteamProcessor
from process.yelp_processor import YelpProcessor


def combine_config(config: dict, **kwargs):
    for k, v in kwargs.items():
        if k not in config:
            config[k] = v
    return config


def seeding(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # tensorflow.random.set_seed(seed)


def argparse():
    arguments = sys.argv[1:]
    kwargs = {}

    key = None
    for arg in arguments:
        if key is not None:
            kwargs[key] = arg
            key = None
        else:
            assert arg.startswith('--')
            key = arg[2:]

    for key, value in kwargs.items():
        if value == 'null':
            kwargs[key] = None
        elif value.isdigit():
            kwargs[key] = int(value)
        elif value.lower() == 'true':
            kwargs[key] = True
        elif value.lower() == 'false':
            kwargs[key] = False
        else:
            try:
                kwargs[key] = float(value)
            except ValueError:
                pass
    return kwargs


def load_processor(dataset, use_cache=True, data_dir=None):
    processors = [MINDProcessor, MicroLensProcessor, SteamProcessor, YelpProcessor, GoodreadsProcessor, MovieLensProcessor]

    for processor in processors:
        if processor.get_name() == dataset:
            pnt(f'loading {processor.get_name()} processor')
            return processor(cache=use_cache, data_dir=data_dir)
    raise ValueError(f'Unknown dataset: {dataset}')
