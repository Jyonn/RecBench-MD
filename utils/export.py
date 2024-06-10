import os
from typing import Dict

import numpy as np


class Exporter:
    def __init__(self, path):
        self.path = path
        self.error_path = path + '.progress'
        self.metrics_path = path + '.metrics'
        self.embed_path = path + '.{0}.npy'
        self.convert_path = path + '.convert'

    def write(self, data):
        with open(self.path, 'a') as f:
            f.write(f'{data}\n')

    def read(self, from_convert=False, to_float=True):
        path = self.convert_path if from_convert else self.path
        handler = float if to_float else str
        # each line is a float
        with open(path, 'r') as f:
            return [handler(line.strip()) for line in f]

    def save_progress(self, index):
        with open(self.error_path, 'w') as f:
            f.write(str(index))

    def load_progress(self):
        if not os.path.exists(self.error_path):
            return 0

        with open(self.error_path, 'r') as f:
            return int(f.read())

    def save_metrics(self, metrics):
        # for metric, value in results.items():
        #     pnt(f'{metric}: {value:.4f}')
        with open(self.metrics_path, 'w') as f:
            for metric, value in metrics.items():
                f.write(f'{metric}: {value:.4f}\n')

    def load_embed(self, entity):
        if not os.path.exists(self.embed_path.format(entity)):
            return {}
        return np.load(self.embed_path.format(entity), allow_pickle=True).item()

    def save_embed(self, entity, embed_dict: Dict[str, np.ndarray]):
        np.save(self.embed_path.format(entity), embed_dict, allow_pickle=True)

    def save_convert(self, scores):
        with open(self.convert_path, 'w') as f:
            for score in scores:
                f.write(f'{score}\n')
