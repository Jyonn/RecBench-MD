import os.path
import random
from typing import Union, Callable

import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from process.base_processor import BaseProcessor


class BaseSeqProcessor(BaseProcessor):
    BASE_STORE_DIR = 'seqdata'

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('SeqProcessor', '').lower()

    def compress(self):
        if self.meta.compressed:
            return False

        item_set = set()
        old_item_size = len(self.items)
        self.users[self.HIS_COL].apply(lambda x: [item_set.add(i) for i in x])
        self.items = self.items[self.items[self.IID_COL].isin(item_set)]
        pnt(f'compressed items from {old_item_size} to {len(self.items)}')

        self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))

        self.meta.compressed = True
        self.meta.save()
        return True

    def load(self):
        if os.path.exists(os.path.join(self.store_dir, 'items.parquet')) and \
                os.path.exists(os.path.join(self.store_dir, 'users.parquet')):
            pnt(f'loading {self.get_name()} from cache')
            self.items = pd.read_parquet(os.path.join(self.store_dir, 'items.parquet'))
            pnt(f'loaded {len(self.items)} items')
            self.users = pd.read_parquet(os.path.join(self.store_dir, 'users.parquet'))
            pnt(f'loaded {len(self.users)} users')

            self.items = self._stringify(self.items)
            self.users = self._stringify(self.users)
        else:
            pnt(f'loading {self.get_name()} from raw data')
            self.items = self.load_items()
            self.items = self._stringify(self.items)
            pnt(f'loaded {len(self.items)} items')
            self.users = self.load_users()
            self.users = self._stringify(self.users)
            pnt(f'loaded {len(self.users)} users')

            self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))
            self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))

        if self.REQUIRE_STRINGIFY:
            self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(
                lambda x: [str(item) for item in x]
            )

        self.item_vocab = dict(zip(self.items[self.IID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.UID_COL], range(len(self.users))))

        if self.compress():
            pnt(f'compressed {self.get_name()} data, re-run to load compressed data')
            return self.load()

        self.load_public_sets()
        return self

    def _iterate(
            self,
            dataframe: pd.DataFrame,
            slicer: Union[int, Callable],
            **kwargs,
    ):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)

        for _, row in dataframe.iterrows():
            uid = row[self.UID_COL]
            history = slicer(row[self.HIS_COL])

            yield uid, history

    def get_source_set(self, source):
        assert source in ['test', 'finetune', 'original'], 'source must be test, finetune, or original'
        return self.users if source == 'original' else getattr(self, f'{source}_set')

    def generate(
            self,
            slicer: Union[int, Callable],
            source='test',
            **kwargs,
    ):
        """
        generate test, finetune, or original set
        :param slicer: user sequence slicer
        :param source: test, finetune, or original
        """
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        source_set = self.get_source_set(source)
        return self._iterate(source_set, slicer)

    def iterate(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='original')

    def test(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='test')

    def finetune(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='finetune')

    def _iterator(self, user_order, users):
        for uid in user_order:
            user = users[users[self.UID_COL] == uid]
            yield user.iloc[0]

    def _split(self, iterator, count):
        users = []
        for user in tqdm(iterator, total=count):  # type: pd.Series
            users.append(user)
            if len(users) >= count:
                break
        users = pd.DataFrame(users)
        return users

    def _load_user_order(self):
        # check if user order exists
        path = os.path.join(self.store_dir, 'user_order.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return [line.strip() for line in f]

        users = self.users[self.UID_COL].unique().tolist()
        random.shuffle(users)
        # save user order
        with open(path, 'w') as f:
            for u in users:
                f.write(f'{u}\n')

        return users

    def load_public_sets(self):
        if self.test_set_valid and self.finetune_set_valid:
            pnt(f'loading {self.get_name()} from cache')

            if self.NUM_TEST:
                self.test_set = pd.read_parquet(os.path.join(self.store_dir, 'test.parquet'))
                self.test_set = self._stringify(self.test_set)
                pnt('loaded test set')

            if self.NUM_FINETUNE:
                self.finetune_set = pd.read_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
                self.finetune_set = self._stringify(self.finetune_set)
                pnt('loaded finetune set')

            self._loaded = True
            return

        pnt(f'processing {self.get_name()} from item and user data')

        users_order = self._load_user_order()
        iterator = self._iterator(users_order, self.users)

        if self.NUM_TEST:
            self.test_set = self._split(iterator, self.NUM_TEST)
            print(self.test_set)
            self.test_set.reset_index(drop=True, inplace=True)
            print(self.test_set)
            self.test_set.to_parquet(os.path.join(self.store_dir, 'test.parquet'))
            pnt(f'generated test set with {len(self.test_set)}/{self.NUM_TEST} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self._split(iterator, self.NUM_FINETUNE)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.finetune_set.to_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
            pnt(f'generated finetune set with {len(self.finetune_set)}/{self.NUM_FINETUNE} samples')

        self._loaded = True
