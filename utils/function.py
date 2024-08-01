import os
import random
import sys

import numpy as np
import torch
from pigmento import pnt

from loader.class_hub import ClassHub


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
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
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


def load_processor(dataset, data_dir=None):
    processors = ClassHub.processors()
    if dataset not in processors:
        raise ValueError(f'Unknown dataset: {dataset}')
    processor = processors[dataset]
    pnt(f'loading {processor.get_name()} processor')
    return processor(data_dir=data_dir)


def load_seq_processor(dataset, data_dir=None):
    processors = ClassHub.seq_processors()
    if dataset not in processors:
        raise ValueError(f'Unknown dataset: {dataset}')
    processor = processors[dataset]
    pnt(f'loading {processor.get_name()} processor')
    return processor(data_dir=data_dir)
