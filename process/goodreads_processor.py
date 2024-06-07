import os

import pandas as pd
from pigmento import pnt

from process.base_uict_processor import UICTProcessor


class GoodreadsSamplingProcessor(UICTProcessor):
    IID_COL = 'bid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    CLK_COL = 'click'
    DAT_COL = 'date'

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'goodreads_book_works.json')
        items = pd.read_json(path, lines=True)
        items = items[['best_book_id', 'original_title']]
        # if original title strip is empty, then skip
        items = items[items['original_title'].str.strip() != '']
        items.columns = [self.IID_COL, 'title']
        return items

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'goodreads_interactions_dedup.json')
        interactions = pd.read_json(path, lines=True)
        pnt('interaction loaded')
        interactions = interactions[['user_id', 'book_id', 'is_read', 'date_added']]

        interactions = interactions[interactions['book_id'].isin(item_set)]
        interactions['is_read'] = interactions['is_read'].astype(int)
        interactions.columns = [self.UID_COL, self.IID_COL, self.CLK_COL, self.DAT_COL]

        return self._load_users(interactions)
