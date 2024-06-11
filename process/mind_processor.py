import os
from typing import cast

import pandas as pd

from process.base_processor import BaseProcessor


class MINDProcessor(BaseProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    CLK_COL = 'click'

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'train', 'news.tsv')
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=['nid', 'cat', 'subcat', 'title', 'abs', 'url', 'tit_ent', 'abs_ent'],
            usecols=['nid', 'cat', 'subcat', 'title', 'abs'],
        )

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'train', 'behaviors.tsv')
        users = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=['imp', 'uid', 'time', 'history', 'predict'],
            usecols=['uid', 'history']
        )

        users['history'] = users['history'].str.split()

        # remove users without history
        users = users.dropna(subset=['history'])
        users = users[users['history'].map(lambda x: len(x) > 0)]

        return users

    def load_interactions(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'train', 'behaviors.tsv')
        interactions = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            names=['imp', 'uid', 'time', 'history', 'predict'],
            usecols=['uid', 'predict']
        )
        interactions['predict'] = interactions['predict'].str.split().apply(
            lambda x: [item.split('-') for item in x]
        )
        interactions = interactions.explode('predict')
        interactions[['nid', 'click']] = pd.DataFrame(interactions['predict'].tolist(), index=interactions.index)
        interactions.drop(columns=['predict'], inplace=True)
        interactions['click'] = interactions['click'].astype(int)
        return interactions
