import hashlib
import os

import pandas as pd
from pigmento import pnt

from loader.code_preparer import CodePreparer
from loader.discrete_code_dataset import DiscreteCodeDataset
from utils.code import get_code_indices


class DiscreteCodePreparer(CodePreparer):
    DATASET_CLASS = DiscreteCodeDataset

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        code_indices, _, _ = get_code_indices(self.conf.code_path)

        self.processor.load()
        self.code_indices = dict()
        self.code_tree = dict()

        item_indices = self.processor.items[self.processor.IID_COL]
        for item_index in item_indices:
            current_indices = code_indices[item_index]
            self.code_indices[item_index] = current_indices
            current_node = self.code_tree
            for index in current_indices:
                if index not in current_node:
                    current_node[index] = dict()
                current_node = current_node[index]

        self.test_datapath = os.path.join(self.store_dir, 'test.parquet')
        self.test_has_generated = os.path.exists(self.test_datapath)

        pnt(f'prepared data will be stored in {self.store_dir}')

    def get_secondary_meta(self):
        return dict(
            code_path=self.conf.code_path,
        )

    def get_secondary_signature(self):
        meta = self.get_secondary_meta()
        keys = sorted(meta.keys())
        key = '-'.join([f'{k}={meta[k]}' for k in keys])
        md5 = hashlib.md5(key.encode()).hexdigest()
        return md5[:6] + f'@{self.conf.valid_ratio}'

    def tokenize_items(self, source='finetune', item_attrs=None):
        return self.code_indices

    def load_datalist(self):
        return self._process()

    def load_or_generate(self, mode='train'):
        if mode == 'test':
            if self.test_has_generated:
                pnt(f'loading prepared {mode} data on {self.processor.get_name()} dataset')
                return self._pack_datalist(pd.read_parquet(self.test_datapath))
            else:
                test_datalist = self._process(source='test')
                test_datalist = pd.DataFrame(test_datalist)
                test_datalist.to_parquet(self.test_datapath)
                return self._pack_datalist(test_datalist)

        return super().load_or_generate(mode)
