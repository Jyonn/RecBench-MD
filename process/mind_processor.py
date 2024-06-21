import os
from typing import cast

import pandas as pd

from process.base_processor import BaseProcessor


class MINDProcessor(BaseProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'

    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'train', 'news.tsv')
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=[self.IID_COL, 'cat', 'subcat', 'title', 'abs', 'url', 'tit_ent', 'abs_ent'],
            usecols=[self.IID_COL, 'cat', 'subcat', 'title', 'abs'],
        )

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'train', 'behaviors.tsv')
        users = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=['imp', self.UID_COL, 'time', self.HIS_COL, 'predict'],
            usecols=[self.UID_COL, self.HIS_COL]
        )

        users[self.HIS_COL] = users[self.HIS_COL].str.split()
        users = users.dropna(subset=[self.HIS_COL])

        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: [item for item in x if item in item_set])
        users = users[users[self.HIS_COL].map(lambda x: len(x) > 0)]

        return users

    def load_interactions(self) -> pd.DataFrame:
        user_set = set(self.users[self.UID_COL].unique())

        path = os.path.join(self.data_dir, 'train', 'behaviors.tsv')
        interactions = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=['imp', self.UID_COL, 'time', self.HIS_COL, 'predict'],
            usecols=[self.UID_COL, 'predict']
        )
        interactions = interactions[interactions[self.UID_COL].isin(user_set)]
        interactions['predict'] = interactions['predict'].str.split().apply(
            lambda x: [item.split('-') for item in x]
        )
        interactions = interactions.explode('predict')
        interactions[[self.IID_COL, self.LBL_COL]] = pd.DataFrame(interactions['predict'].tolist(), index=interactions.index)
        interactions.drop(columns=['predict'], inplace=True)
        interactions[self.LBL_COL] = interactions[self.LBL_COL].astype(int)
        return interactions
