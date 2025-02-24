import os
from typing import cast

import pandas as pd

from process.base_processor import BaseProcessor


class PENSProcessor(BaseProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'

    NUM_TEST = 0
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'news.tsv')
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            header=0,
            names=[self.IID_COL, 'category', 'topic', 'title', 'body', 'entity', 'content'],
            usecols=[self.IID_COL, 'category', 'topic', 'title', 'body'],
        )

    def _load_user(self, mode):
        path = os.path.join(self.data_dir, f'{mode}.tsv')
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            header=0,
            names=[self.UID_COL, 'history', 'dwell_time', 'exposure_time', 'pos', 'neg', 'start', 'end',
                   'dwell_time_pos'],
            usecols=[self.UID_COL, 'history', 'pos', 'neg'],
        )

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        users_train = self._load_user('train')
        users_dev = self._load_user('valid')
        users = pd.concat([users_train, users_dev])
        # reset index
        users = users.reset_index(drop=True)

        users[self.HIS_COL] = users[self.HIS_COL].str.split()
        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: [item for item in x if item in item_set])
        users = users[users[self.HIS_COL].map(lambda x: len(x) > 0)]

        self._interactions = users[[self.UID_COL, 'pos', 'neg']]
        users = users.drop(columns=['pos', 'neg'])
        return users

    def load_interactions(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())
        interactions = []
        for i, row in self._interactions.iterrows():
            uid = row[self.UID_COL]
            pos = list(filter(lambda x: x in item_set, row['pos'].split()))
            neg = list(filter(lambda x: x in item_set, row['neg'].split()))

            if not pos or not neg:
                continue

            for nid in pos:
                interactions.append([uid, nid, 1])

            for nid in neg:
                interactions.append([uid, nid, 0])
        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.LBL_COL])
        return interactions
