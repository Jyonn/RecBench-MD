import abc
import os.path
import pdb
import random

import numpy as np
import pandas as pd
from pigmento import pnt


class BaseProcessor(abc.ABC):
    IID_COL = 'iid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    CLK_COL = 'click'

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    def __init__(self, data_dir=None, cache=True):
        self.data_dir = data_dir
        self.store_dir = os.path.join('data', self.get_name())
        os.makedirs(self.store_dir, exist_ok=True)

        self.cache = cache

        self.items = None
        self.users = None
        self.interactions = None

        self.item_vocab = None
        self.user_vocab = None

        self.test_set = None
        self.finetune_set = None

    @property
    def default_attr(self):
        raise None

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Processor', '').lower()

    def load_items(self) -> pd.DataFrame:
        raise NotImplemented

    def load_users(self) -> pd.DataFrame:
        raise NotImplemented

    def load_interactions(self) -> pd.DataFrame:
        raise NotImplemented

    def load(self):
        # if exists, load from store
        if os.path.exists(os.path.join(self.store_dir, 'items.parquet')) and \
                os.path.exists(os.path.join(self.store_dir, 'users.parquet')) and \
                os.path.exists(os.path.join(self.store_dir, 'interactions.parquet')):
            pnt(f'loading {self.get_name()} from cache')
            self.items = pd.read_parquet(os.path.join(self.store_dir, 'items.parquet'))
            pnt('loaded items')
            self.users = pd.read_parquet(os.path.join(self.store_dir, 'users.parquet'))
            pnt('loaded users')
            self.interactions = pd.read_parquet(os.path.join(self.store_dir, 'interactions.parquet'))
            pnt('loaded interactions')
        else:
            pnt(f'loading {self.get_name()} from raw data')
            self.items = self.load_items()
            pnt('loaded items')
            self.users = self.load_users()
            pnt('loaded users')
            self.interactions = self.load_interactions()
            pnt('loaded interactions')

            if self.cache:
                self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))
                self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))
                self.interactions.to_parquet(os.path.join(self.store_dir, 'interactions.parquet'))

        self.items[self.IID_COL] = self.items[self.IID_COL].astype(str)
        self.users[self.UID_COL] = self.users[self.UID_COL].astype(str)
        self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(
            lambda x: [str(item) for item in x]
        )
        self.interactions[self.IID_COL] = self.interactions[self.IID_COL].astype(str)
        self.interactions[self.UID_COL] = self.interactions[self.UID_COL].astype(str)

        self.item_vocab = dict(zip(self.items[self.IID_COL], self.items.index))
        self.user_vocab = dict(zip(self.users[self.UID_COL], self.users.index))

    def _organize_item(self, iid, item_attrs: list):
        item = self.items.loc[self.item_vocab[iid]]
        if len(item_attrs) == 1:
            return item[item_attrs[0]]

        item_str = []
        for attr in item_attrs:
            item_str.append(f'{attr}: {item[attr]}')
        return ', '.join(item_str)

    def generate(self, max_len, item_attrs=None):
        self.load()

        item_attrs = item_attrs or self.default_attr
        # iterate interactions
        for _, row in self.interactions.iterrows():
            uid = row[self.UID_COL]
            candidate = row[self.IID_COL]
            click = row[self.CLK_COL]

            user = self.users.loc[self.user_vocab[uid]]
            history = user[self.HIS_COL][:max_len]
            history_str = [self._organize_item(iid, item_attrs) for iid in history]
            candidate_str = self._organize_item(candidate, item_attrs)

            yield uid, candidate, history_str, candidate_str, click

    @staticmethod
    def _group_iterator(users, interactions):
        for u in users:
            yield interactions.get_group(u)

    @staticmethod
    def _split(iterator, count):
        df = pd.DataFrame()
        for group in iterator:
            df = df.append(group)
            if len(df) >= count:
                break
        return df

    def load_user_order(self):
        # check if user order exists
        if os.path.exists(os.path.join(self.store_dir, 'user_order.txt')):
            with open(os.path.join(self.store_dir, 'user_order.txt'), 'r') as f:
                return [line.strip() for line in f]

        users = self.interactions[self.UID_COL].unique().tolist()
        random.shuffle(users)
        # save user order
        with open(os.path.join(self.store_dir, 'user_order.txt'), 'w') as f:
            for u in users:
                f.write(f'{u}\n')

        return users

    def process(self):
        if os.path.exists(os.path.join(self.store_dir, 'test.parquet')) and \
                os.path.exists(os.path.join(self.store_dir, 'finetune.parquet')):
            pnt(f'loading {self.get_name()} from cache')

            self.test_set = pd.read_parquet(os.path.join(self.store_dir, 'test.parquet'))
            pnt('loaded test set')
            self.finetune_set = pd.read_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
            pnt('loaded finetune set')
            return

        pnt(f'processing {self.get_name()} from item, user, and interaction data')
        self.load()

        users_order = self.load_user_order()
        interactions = self.interactions.groupby(self.UID_COL)

        iterator = self._group_iterator(users_order, interactions)
        if self.NUM_TEST:
            self.test_set = self._split(iterator, self.NUM_TEST)
            self.test_set.reset_index(drop=True, inplace=True)
            self.test_set.to_parquet(os.path.join(self.store_dir, 'test.parquet'))
            pnt(f'generated test set with {len(self.test_set)}/{self.NUM_TEST} samples')

        if self.NUM_FINETUNE:
            self.finetune_set = self._split(iterator, self.NUM_FINETUNE)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.finetune_set.to_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
            pnt(f'generated finetune set with {len(self.finetune_set)}/{self.NUM_FINETUNE} samples')
