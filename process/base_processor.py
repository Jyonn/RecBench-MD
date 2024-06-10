import abc
import os.path
import random
from typing import Union, Callable, Optional

import pandas as pd
from pigmento import pnt


class BaseProcessor(abc.ABC):
    IID_COL: str
    UID_COL: str
    HIS_COL: str
    CLK_COL: str

    NUM_TEST: int
    NUM_FINETUNE: int

    MAX_INTERACTIONS_PER_USER: int = 20
    REQUIRE_STRINGIFY: bool

    def __init__(self, data_dir=None, cache=True):
        self.data_dir = data_dir
        self.store_dir = os.path.join('data', self.get_name())
        os.makedirs(self.store_dir, exist_ok=True)

        self.cache: bool = cache

        self._loaded: bool = False

        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None

        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

        self.test_set: Optional[pd.DataFrame] = None
        self.finetune_set: Optional[pd.DataFrame] = None

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
        if not self.REQUIRE_STRINGIFY:
            return df
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
            pnt(f'loaded {len(self.items)} items')
            self.users = pd.read_parquet(os.path.join(self.store_dir, 'users.parquet'))
            pnt(f'loaded {len(self.users)} users')
            self.interactions = pd.read_parquet(os.path.join(self.store_dir, 'interactions.parquet'))
            pnt(f'loaded {len(self.interactions)} interactions')
        else:
            pnt(f'loading {self.get_name()} from raw data')
            self.items = self.load_items()
            pnt(f'loaded {len(self.items)} items')
            self.users = self.load_users()
            pnt(f'loaded {len(self.users)} users')
            self.interactions = self.load_interactions()
            pnt(f'loaded {len(self.interactions)} interactions')

            if self.cache:
                self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))
                self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))
                self.interactions.to_parquet(os.path.join(self.store_dir, 'interactions.parquet'))

        self.items = self._stringify(self.items)
        self.users = self._stringify(self.users)
        self.interactions = self._stringify(self.interactions)

        if self.REQUIRE_STRINGIFY:
            self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(
                lambda x: [str(item) for item in x]
            )

        self.item_vocab = dict(zip(self.items[self.IID_COL], self.items.index))
        self.user_vocab = dict(zip(self.users[self.UID_COL], self.users.index))

        self.load_public_sets()
        # self._loaded = True

    def _organize_item(self, iid, item_attrs: list):
        item = self.items.loc[self.item_vocab[iid]]
        if len(item_attrs) == 1:
            return item[item_attrs[0]]

        item_str = []
        for attr in item_attrs:
            item_str.append(f'{attr}: {item[attr]}')
        return ', '.join(item_str)

    @staticmethod
    def _build_slicer(slicer: int):
        def _slicer(x):
            return x[:slicer] if slicer > 0 else x[slicer:]
        return _slicer

    def _iterate(self, dataframe: pd.DataFrame, slicer: Union[int, Callable], item_attrs=None):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)
        item_attrs = item_attrs or self.default_attrs

        for _, row in dataframe.iterrows():
            uid = row[self.UID_COL]
            candidate = row[self.IID_COL]
            click = row[self.CLK_COL]

            user = self.users.loc[self.user_vocab[uid]]
            history = slicer(user[self.HIS_COL])
            history_str = [self._organize_item(iid, item_attrs) for iid in history]
            candidate_str = self._organize_item(candidate, item_attrs)

            yield uid, candidate, history_str, candidate_str, click

    def get_source_set(self, source):
        assert source in ['test', 'finetune', 'original'], 'source must be test, finetune, or original'
        return self.interactions if source == 'original' else getattr(self, f'{source}_set')

    def generate(self, slicer: Union[int, Callable], item_attrs=None, source='test'):
        """
        generate test, finetune, or original set
        :param slicer: user sequence slicer
        :param item_attrs: item attributes to show
        :param source: test, finetune, or original
        """
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        source_set = self.get_source_set(source)
        return self._iterate(source_set, slicer, item_attrs)

    def iterate(self, slicer: Union[int, Callable], item_attrs=None):
        return self.generate(slicer, item_attrs, source='original')

    def test(self, slicer: Union[int, Callable], item_attrs=None):
        return self.generate(slicer, item_attrs, source='test')

    def finetune(self, slicer: Union[int, Callable], item_attrs=None):
        return self.generate(slicer, item_attrs, source='finetune')

    @staticmethod
    def _group_iterator(users, interactions):
        for u in users:
            yield interactions.get_group(u)

    def _split(self, iterator, count):
        df = pd.DataFrame()
        for group in iterator:
            for click in range(2):
                group_click = group[group[self.CLK_COL] == click]
                selected_group_click = group_click.sample(n=min(self.MAX_INTERACTIONS_PER_USER // 2, len(group_click)), replace=False)
                df = pd.concat([df, selected_group_click])
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

            self._loaded = True
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

        self._loaded = True
