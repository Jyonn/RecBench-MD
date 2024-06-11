import abc
import os.path
import random

import pandas as pd
from pigmento import pnt


class BaseProcessor(abc.ABC):
    IID_COL: str
    UID_COL: str
    HIS_COL: str
    CLK_COL: str

    NUM_TEST: int
    NUM_FINETUNE: int

    def __init__(self, data_dir=None, cache=True):
        self.data_dir = data_dir
        self.store_dir = os.path.join('data', self.get_name())
        os.makedirs(self.store_dir, exist_ok=True)

        self.cache = cache

        self._loaded = False

        self.items = None
        self.users = None
        self.interactions = None

        self.item_vocab = None
        self.user_vocab = None

        self.test_set = None
        self.finetune_set = None

    @property
    def default_attrs(self):
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

    def _stringify(self, df: pd.DataFrame):
        if self.IID_COL in df.columns:
            df[self.IID_COL] = df[self.IID_COL].astype(str)
        if self.UID_COL in df.columns:
            df[self.UID_COL] = df[self.UID_COL].astype(str)
        return df

    def load(self):
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

        self.items = self._stringify(self.items)
        self.users = self._stringify(self.users)
        self.interactions = self._stringify(self.interactions)

        self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(
            lambda x: [str(item) for item in x]
        )

        self.item_vocab = dict(zip(self.items[self.IID_COL], self.items.index))
        self.user_vocab = dict(zip(self.users[self.UID_COL], self.users.index))

        self.load_public_sets()

        self._loaded = True

    def _organize_item(self, iid, item_attrs: list):
        # import pdb
        # pdb.set_trace()
        item = self.items.loc[self.item_vocab[iid]]
        if len(item_attrs) == 1:
            return item[item_attrs[0]]

        item_str = []
        for attr in item_attrs:
            item_str.append(f'{attr}: {item[attr]}')
        return ', '.join(item_str)

    def iterate(self, max_len=10, item_attrs=None):
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        item_attrs = item_attrs or self.default_attrs
        # iterate interactions
        for _, row in self.interactions.iterrows():
            uid = row[self.UID_COL]
            candidate = row[self.IID_COL]
            click = row[self.CLK_COL]

            user = self.users.loc[self.user_vocab[uid]]
            history = user[self.HIS_COL][:max_len]
            # import pdb
            # pdb.set_trace()
            # if(len(history)==1):# 
            #     history = [item for sublist in history for item in sublist.split()]
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
            df =pd.concat([df,group]) #df.append(group)
            if len(df) >= count:
                break
        return df

    def _load_user_order(self):
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

    def load_public_sets(self):
        if os.path.exists(os.path.join(self.store_dir, 'test.parquet')) and \
                os.path.exists(os.path.join(self.store_dir, 'finetune.parquet')):
            pnt(f'loading {self.get_name()} from cache')

            if self.NUM_TEST:
                self.test_set = pd.read_parquet(os.path.join(self.store_dir, 'test.parquet'))
                self.test_set = self._stringify(self.test_set)
                pnt('loaded test set')

            if self.NUM_FINETUNE:
                self.finetune_set = pd.read_parquet(os.path.join(self.store_dir, 'finetune.parquet'))
                self.finetune_set = self._stringify(self.finetune_set)
                pnt('loaded finetune set')

            return

        pnt(f'processing {self.get_name()} from item, user, and interaction data')

        users_order = self._load_user_order()
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