import os

import pandas as pd
from process.base_ns_processor import NSProcessor
from process.base_uspe_processor import USPEProcessor


class MicroLensSamplingProcessor(NSProcessor, USPEProcessor):
    IID_COL = 'item'
    UID_COL = 'user'
    HIS_COL = 'history'
    CLK_COL = 'click'

    POS_COUNT = 2
    NEG_RATIO = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

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

        self._get_user_dict_from_interactions(interactions)

        users = interactions.sort_values(
            [self.UID_COL, 'ts']
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        self._extract_pos_samples(users)

        return users
