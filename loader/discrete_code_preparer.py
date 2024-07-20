import hashlib

from pigmento import pnt
from tqdm import tqdm

from loader.code_preparer import CodePreparer
from loader.discrete_code_dataset import DiscreteCodeDataset
from loader.code_map import CodeMap as Map
from loader.token_vocab import TV
from utils.code import get_code_indices


class DiscreteCodePreparer(CodePreparer):
    DATASET_CLASS = DiscreteCodeDataset

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.code_indices, _ = get_code_indices(self.conf.code_path)

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
