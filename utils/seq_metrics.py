from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, Union, List

import numpy as np
import pandas as pd
import torch


class SeqMetric:
    name: str
    group: bool
    minimize: bool

    def __init__(self, num_items, prod_mode):
        self.num_items = num_items
        self.prod_mode = prod_mode

    def _calculate(self, rank: int) -> Union[int, float]:
        raise NotImplementedError

    def calculate(self, ranks: list) -> Union[int, float]:
        scores = [self._calculate(rank) for rank in ranks]
        if self.prod_mode:
            return np.prod(scores)
        return np.mean(scores)

    def __call__(self, *args, **kwargs) -> Union[int, float]:
        return self.calculate(*args, **kwargs)

    def __str__(self):
        return self.name


class AUC(SeqMetric):
    name = 'AUC'
    group = False
    minimize = False

    def _calculate(self, rank: int):
        if rank < 0:
            return 0.0
        num_lower_ranked = self.num_items - rank
        return (num_lower_ranked + 1.0) / self.num_items


class GAUC(AUC):
    name = 'GAUC'
    group = True
    minimize = False


class MRR(SeqMetric):
    name = 'MRR'
    group = True
    minimize = False

    def _calculate(self, rank: int):
        if rank < 0:
            return 0.0
        return 1.0 / rank


class HitRatio(SeqMetric):
    name = 'HitRatio'
    group = True
    minimize = False

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def _calculate(self, rank: int):
        if rank < 0:
            return 0
        return int(rank <= self.n)

    def __str__(self):
        return f'{self.name}@{self.n}'


class Recall(HitRatio):
    name = 'Recall'


class NDCG(SeqMetric):
    name = 'NDCG'
    group = True
    minimize = False

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def _calculate(self, rank: int):
        if self.prod_mode:
            if rank < 1:
                return 0
            return 1.0 / np.log2(rank + 1)
        if rank < 1 or rank > self.n:
            return 0
        return 1.0 / np.log2(rank + 1)

    def __str__(self):
        return f'{self.name}@{self.n}'


class SeqMetricPool:
    metric_list = [AUC, GAUC, Recall, NDCG, HitRatio, MRR]
    metric_dict = {m.name.upper(): m for m in metric_list}

    def __init__(self, metrics):
        self.metrics = metrics  # type: List[SeqMetric]
        self.values = OrderedDict()  # type: Dict[str, Union[list, float]]
        self.group = False

        for metric in self.metrics:
            self.values[str(metric)] = []
            self.group = self.group or metric.group

    @classmethod
    def parse(cls, metrics_config, num_items, prod_mode):
        metrics = []
        for m in metrics_config:
            at = m.find('@')
            argument = []
            if at > -1:
                m, argument = m[:at], [int(m[at+1:])]
            if m.upper() not in SeqMetricPool.metric_dict:
                raise ValueError(f'Metric {m} not found')
            metrics.append(SeqMetricPool.metric_dict[m.upper()](num_items=num_items, prod_mode=prod_mode, *argument))
        return cls(metrics)

    def calculate(self, ranks, groups, group_worker=5):
        if not self.metrics:
            return {}

        df = pd.DataFrame(dict(groups=groups, ranks=ranks))

        groups = None
        if self.group:
            groups = df.groupby('groups')

        for metric in self.metrics:
            if not metric.group:
                self.values[str(metric)] = metric(
                    scores=ranks,
                )
                continue

            tasks = []
            pool = Pool(processes=group_worker)

            for g in groups:
                group = g[1]
                g_ranks = group.ranks.tolist()
                tasks.append(pool.apply_async(metric, args=(g_ranks,)))

            pool.close()
            pool.join()
            values = [t.get() for t in tasks]
            self.values[str(metric)] = torch.tensor(values, dtype=torch.float).mean().item()
        return self.values

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    @classmethod
    def is_minimize(cls, metric: str):
        if isinstance(metric, SeqMetric):
            return metric.minimize
        assert isinstance(metric, str)
        metric = metric.split('@')[0]
        return cls.metric_dict[metric].minimize
