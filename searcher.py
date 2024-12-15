"""
search configuration in tuning/<model> dir, each configuration is a json file
"""
import datetime
import json
import os

import pigmento
from pigmento import pnt
from prettytable import PrettyTable

from utils.config_init import ConfigInit


class Searcher:
    def __init__(self, conf):
        self.conf = conf
        self.model = conf.model
        self.keys = dict(
            mode=conf.mode,
            data=conf.data,
        )
        self._keys = ['mode', 'data']
        for key in conf:
            if key not in ['model', 'attrs', 'mode', 'data']:
                self.keys[key] = conf[key]
                self._keys.append(key)

        self.dir = os.path.join('tuning', self.model)
        os.makedirs(self.dir, exist_ok=True)

        # get all *.json files in the dir
        self.files = [f for f in os.listdir(self.dir) if f.endswith('.json')]

    def search(self):
        candidates = []
        print(f'searching among {len(self.files)} files')
        for file in self.files:
            sign = file.split('.')[0]

            path = os.path.join(self.dir, file)
            # get create time of the file
            create_time = os.path.getctime(path)

            try:
                conf = json.load(open(path, 'r'))
                _, _ = conf['mode'], conf['data']
            except:
                continue

            match = True
            for key in self._keys:
                if self.keys[key] is not None and key not in conf:
                    match = False
                    break
            if not match:
                continue

            candidates.append((sign, conf, create_time))

        return self.display(candidates)

    def display(self, candidates):
        # use prettytable to display the candidates
        filtered_candidates = []
        attrs = self.conf.attrs.split('+') if self.conf.attrs else []
        table = PrettyTable()
        table.field_names = ['#', 'sign', 'time', *self._keys, *attrs, 'conf', 'log']
        for index, (sign, conf, create_time) in enumerate(candidates):
            match = True
            for key in self._keys:
                if key not in conf or (self.keys[key] is not None and self.keys[key] not in conf[key]):
                    match = False
                    break
            if not match:
                continue
            # make float time to human-readable time using time or datetime module
            # time format: Oct. 31, 12:33
            dt = datetime.datetime.fromtimestamp(create_time).strftime('%b. %d, %H:%M')
            filtered_candidates.append((sign, conf, dt))
            row = [index, sign, dt]
            for key in self._keys:
                row.append(conf[key])
            for attr in attrs:
                row.append(conf.get(attr, None))
            row.append(f'cat {self.dir}/{sign}.json')
            row.append(f'cat {self.dir}/{sign}.log')
            table.add_row(row)

        print(table)
        return filtered_candidates

    @staticmethod
    def compare(candidates):
        indices = input('Indices of configurations you want to compare (enter "exit" to exit): ')
        if indices == 'exit':
            exit(0)
        indices = map(lambda x: int(x) if x.isdigit() else -1, indices.split(','))
        indices = list(filter(lambda x: 0 <= x < len(candidates), indices))
        attrs = set()
        for index in indices:
            sign, conf = candidates[index]
            attrs.update(conf.keys())

        different_attrs = []
        for attr in attrs:
            value = candidates[indices[0]][1].get(attr, None)
            for index in indices:
                sign, conf = candidates[index]
                if conf.get(attr, None) != value:
                    different_attrs.append(attr)
                    break

        table = PrettyTable()
        table.field_names = ['#', 'sign', *different_attrs]
        for index in indices:
            sign, conf = candidates[index]
            row = [index, sign]
            for attr in different_attrs:
                row.append(conf.get(attr, None))
            table.add_row(row)

        print(table)


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['model'],
        default_args=dict(
            mode=None,
            data=None,
            attrs=None,
        ),
        makedirs=[]
    ).parse()

    searcher = Searcher(conf=configuration)
    candidates = searcher.search()
    while True:
        searcher.compare(candidates)
