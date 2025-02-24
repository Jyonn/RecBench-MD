import os

import pandas as pd
from tqdm import tqdm

from process.base_processor import BaseProcessor


class EBNeRDProcessor(BaseProcessor):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'

    NUM_TEST = 20_000
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'articles.parquet')
        items = pd.read_parquet(path)
        items = items[['article_id', 'title', 'subtitle', 'body', 'category_str']]
        items.columns = [self.IID_COL, 'title', 'subtitle', 'body', 'category']
        return items

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'train', 'history.parquet')
        users = pd.read_parquet(path)
        users = users[['user_id', 'article_id_fixed']]
        users.columns = [self.UID_COL, self.HIS_COL]

        users = users.dropna(subset=[self.HIS_COL])
        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: [str(item) for item in x if str(item) in item_set])
        users = users[users[self.HIS_COL].map(lambda x: len(x) > 0)]

        return users

    def load_interactions(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())
        user_set = set(self.users[self.UID_COL].unique())

        columns = ['impression_id', 'user_id', 'article_ids_inview', 'article_ids_clicked']
        path = os.path.join(self.data_dir, 'train', 'behaviors.parquet')
        interactions = pd.read_parquet(path, columns=columns)
        interactions = interactions[['user_id', 'article_ids_inview', 'article_ids_clicked']]
        interactions.columns = [self.UID_COL, 'interactions', 'click']

        interactions = interactions[interactions[self.UID_COL].map(lambda x: str(x) in user_set)]
        # interactions['interactions'] = interactions['interactions'].apply(lambda x: [str(item) for item in x if str(item) in item_set])
        # interactions['click'] = interactions['click'].apply(lambda x: [str(item) for item in x if str(item) in item_set])

        _interactions = {self.UID_COL: [], self.IID_COL: [], self.LBL_COL: []}
        for _, row in tqdm(interactions.iterrows(), total=len(interactions)):
            user = str(row[self.UID_COL])
            for item in row['interactions']:
                if str(item) not in item_set:
                    continue
                _interactions[self.UID_COL].append(user)
                _interactions[self.IID_COL].append(str(item))
                _interactions[self.LBL_COL].append(int(item in row['click']))

        interactions = pd.DataFrame(_interactions)
        return interactions
