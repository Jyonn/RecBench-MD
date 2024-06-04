import abc
import os.path

import pandas as pd


class BaseProcessor(abc.ABC):
    IID_COL = 'iid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    CLK_COL = 'click'

    def __init__(self, data_dir, store_dir):
        self.data_dir = data_dir
        self.store_dir = os.path.join(store_dir, self.get_name())
        os.makedirs(self.store_dir, exist_ok=True)

        self.items = None
        self.users = None
        self.interactions = None

        self.item_vocab = None
        self.user_vocab = None

    def get_name(self):
        return self.__class__.__name__.replace('Processor', '').lower()

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
            print(f'loading {self.get_name()} from cache')
            self.items = pd.read_parquet(os.path.join(self.store_dir, 'items.parquet'))
            print('loaded items')
            self.users = pd.read_parquet(os.path.join(self.store_dir, 'users.parquet'))
            print('loaded users')
            self.interactions = pd.read_parquet(os.path.join(self.store_dir, 'interactions.parquet'))
            print('loaded interactions')
        else:
            print(f'loading {self.get_name()} from raw data')
            self.items = self.load_items()
            print('loaded items')
            self.users = self.load_users()
            print('loaded users')
            self.interactions = self.load_interactions()
            print('loaded interactions')

            self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))
            self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))
            self.interactions.to_parquet(os.path.join(self.store_dir, 'interactions.parquet'))

        self.item_vocab = dict(zip(self.items[self.IID_COL], self.items.index))
        self.user_vocab = dict(zip(self.users[self.UID_COL], self.users.index))

    def organize_item(self, iid, item_attrs: list):
        item = self.items.loc[self.item_vocab[iid]]
        if len(item_attrs) == 1:
            return item[item_attrs[0]]

        item_str = []
        for attr in item_attrs:
            item_str.append(f'{attr}: {item[attr]}')
        return ', '.join(item_str)

    def generate(self, max_len, item_attrs):
        self.load()

        # iterate interactions
        for _, row in self.interactions.iterrows():
            uid = row[self.UID_COL]
            candidate = row[self.IID_COL]
            click = row[self.CLK_COL]

            user = self.users.loc[self.user_vocab[uid]]
            history = user[self.HIS_COL][:max_len]
            history_str = [self.organize_item(iid, item_attrs) for iid in history]
            candidate_str = self.organize_item(candidate, item_attrs)

            yield uid, candidate, history_str, candidate_str, click
