import os

import pandas as pd
from process.neg_processor import NegProcessor


class MicroLensProcessor(NegProcessor):
    IID_COL = 'item'
    UID_COL = 'user'
    HIS_COL = 'history'
    CLK_COL = 'click'

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'MicroLens-50k_titles.csv')
        titles_df = pd.read_csv(filepath_or_buffer=path)
        return titles_df[[self.IID_COL, 'title']]

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'MicroLens-50k_pairs.tsv')
        interactions = pd.read_csv(
            filepath_or_buffer=path,
            sep='\t',
            header=None,
            names=[self.UID_COL, self.IID_COL, 'ts']
        )
        interactions = self._stringify(interactions)

        self.user_dict = dict()
        for idx, row in interactions.iterrows():
            user_id = row[self.UID_COL]
            item_id = row[self.IID_COL]
            if user_id not in self.user_dict:
                self.user_dict[user_id] = set()
            self.user_dict[user_id].add(item_id)

        users = interactions.sort_values(
            [self.UID_COL, 'timestamp']
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()

        users.columns = [self.UID_COL, self.HIS_COL]
        users = self._generate_pos_inters_from_history(users)
        return users
