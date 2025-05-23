import abc
import json
import os.path
import random
from typing import Union, Callable, Optional

import pandas as pd
from oba import Obj
from pigmento import pnt


class Meta:
    VER = 'v1.0'

    def __init__(self, path):
        self.path = path

        if not os.path.exists(path):
            self.compressed = False
            self.version = self.VER
        else:
            data = json.load(open(path, 'r'))
            data = Obj(data)
            self.compressed = data.fully_compressed
            self.version = data.version

    def save(self):
        data = {
            'fully_compressed': self.compressed,
            'version': self.version,
        }
        json.dump(data, open(self.path, 'w'))


class BaseProcessor(abc.ABC):
    IID_COL: str
    UID_COL: str
    HIS_COL: str
    LBL_COL: str

    NUM_TEST: int
    NUM_FINETUNE: int

    MAX_HISTORY_PER_USER: int = 100
    MAX_INTERACTIONS_PER_USER: int = 20
    REQUIRE_STRINGIFY: bool

    BASE_STORE_DIR = 'data'

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.store_dir = os.path.join(self.BASE_STORE_DIR, self.get_name())
        os.makedirs(self.store_dir, exist_ok=True)

        self.meta = Meta(os.path.join(self.store_dir, 'meta.json'))

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

    def compress(self):
        if self.meta.compressed:
            return False

        user_set = set(self.interactions[self.UID_COL].unique())
        old_user_size = len(self.users)
        self.users = self.users[self.users[self.UID_COL].isin(user_set)]
        self.users = self.users.drop_duplicates(subset=[self.UID_COL])
        pnt(f'compressed users from {old_user_size} to {len(self.users)}')

        item_set = set(self.interactions[self.IID_COL].unique())
        old_item_size = len(self.items)
        self.users[self.HIS_COL].apply(lambda x: [item_set.add(i) for i in x])
        self.items = self.items[self.items[self.IID_COL].isin(item_set)].reset_index(drop=True)
        pnt(f'compressed items from {old_item_size} to {len(self.items)}')

        self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))
        self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))

        self.meta.compressed = True
        self.meta.save()
        return True

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

            self.items = self._stringify(self.items)
            self.users = self._stringify(self.users)
            self.interactions = self._stringify(self.interactions)
        else:
            pnt(f'loading {self.get_name()} from raw data')
            self.items = self.load_items()
            self.items = self._stringify(self.items)
            pnt(f'loaded {len(self.items)} items')
            self.users = self.load_users()
            self.users = self._stringify(self.users)
            pnt(f'loaded {len(self.users)} users')
            self.interactions = self.load_interactions()
            self.interactions = self._stringify(self.interactions)
            pnt(f'loaded {len(self.interactions)} interactions')

            self.items.to_parquet(os.path.join(self.store_dir, 'items.parquet'))
            self.users.to_parquet(os.path.join(self.store_dir, 'users.parquet'))
            self.interactions.to_parquet(os.path.join(self.store_dir, 'interactions.parquet'))

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

    def organize_item(self, iid, item_attrs: list, as_dict=False, item_self=False):
        if item_self:
            item = iid
        else:
            item = self.items.iloc[self.item_vocab[iid]]

        if as_dict:
            return {attr: item[attr] or '' for attr in item_attrs}

        if len(item_attrs) == 1:
            return item[item_attrs[0]]

        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])

    @staticmethod
    def _build_slicer(slicer: int):
        def _slicer(x):
            return x[:slicer] if slicer > 0 else x[slicer:]
        return _slicer

    def _iterate(
            self,
            dataframe: pd.DataFrame,
            slicer: Union[int, Callable],
            item_attrs=None,
            id_only=False,
            as_dict=False,
    ):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)
        item_attrs = item_attrs or self.default_attrs

        for _, row in dataframe.iterrows():
            uid = row[self.UID_COL]
            candidate = row[self.IID_COL]
            label = row[self.LBL_COL]

            user = self.users.iloc[self.user_vocab[uid]]
            history = slicer(user[self.HIS_COL])

            if id_only:
                yield uid, candidate, history, label
                continue

            history_str = [self.organize_item(iid, item_attrs, as_dict=as_dict) for iid in history]
            candidate_str = self.organize_item(candidate, item_attrs, as_dict=as_dict)

            yield uid, candidate, history_str, candidate_str, label

    def get_source_set(self, source):
        assert source in ['test', 'finetune', 'original'], 'source must be test, finetune, or original'
        return self.interactions if source == 'original' else getattr(self, f'{source}_set')

    def generate(
            self,
            slicer: Union[int, Callable],
            item_attrs=None,
            source='test',
            id_only=False,
            as_dict=False,
            filter_func=None,
    ):
        """
        generate test, finetune, or original set
        :param slicer: user sequence slicer
        :param item_attrs: item attributes to show
        :param source: test, finetune, or original
        :param id_only: whether to return only ids
        :param as_dict: whether to return item attributes as dict or string
        :param filter_func: filter function to apply on the source set
        """
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        source_set = self.get_source_set(source)
        if filter_func:
            source_set = filter_func(source_set)
        return self._iterate(source_set, slicer, item_attrs, id_only=id_only, as_dict=as_dict)

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
            for label in range(2):
                group_lbl = group[group[self.LBL_COL] == label]
                selected_group_lbl = group_lbl.sample(n=min(self.MAX_INTERACTIONS_PER_USER // 2, len(group_lbl)), replace=False)
                df = pd.concat([df, selected_group_lbl])
            if len(df) >= count:
                break
        return df

    def _load_user_order(self):
        # check if user order exists
        path = os.path.join(self.store_dir, 'user_order.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return [line.strip() for line in f]

        users = self.interactions[self.UID_COL].unique().tolist()
        random.shuffle(users)
        # save user order
        with open(path, 'w') as f:
            for u in users:
                f.write(f'{u}\n')

        return users

    def load_valid_user_set(self, valid_ratio: float) -> set:
        path = os.path.join(self.store_dir, f'valid_user_set_{valid_ratio}.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return {line.strip() for line in f}

        users = self.finetune_set[self.UID_COL].unique().tolist()
        random.shuffle(users)

        valid_user_num = int(valid_ratio * len(users))
        valid_user_set = users[:valid_user_num]

        with open(path, 'w') as f:
            for u in valid_user_set:
                f.write(f'{u}\n')

        return set(valid_user_set)

    @property
    def test_set_required(self):
        return self.NUM_TEST > 0

    @property
    def finetune_set_required(self):
        return self.NUM_FINETUNE > 0

    @property
    def test_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'test.parquet')) or not self.test_set_required

    @property
    def finetune_set_valid(self):
        return os.path.exists(os.path.join(self.store_dir, 'finetune.parquet')) or not self.finetune_set_required

    def load_public_sets(self):
        # if os.path.exists(os.path.join(self.store_dir, 'test.parquet')) and \
        #         os.path.exists(os.path.join(self.store_dir, 'finetune.parquet')):
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

    def get_item_subset(self, source, slicer: Union[int, Callable]):
        item_set = set()

        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)

        source_set = self.get_source_set(source)
        for _, row in source_set.iterrows():
            uid = row[self.UID_COL]
            iid = row[self.IID_COL]

            user = self.users.iloc[self.user_vocab[uid]]
            history = slicer(user[self.HIS_COL])

            item_set.add(iid)
            item_set.update(history)

        return item_set
