import os

import pandas as pd
from process.base_ns_processor import NSProcessor
from process.base_uspe_processor import USPEProcessor


class MicroLensProcessor(NSProcessor, USPEProcessor):
    IID_COL = 'item'
    UID_COL = 'user'
    HIS_COL = 'history'
    LBL_COL = 'click'

    POS_COUNT = 2
    NEG_RATIO = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'MicroLens-50k_titles.csv')
        titles_df = pd.read_csv(
            filepath_or_buffer=path,
            sep=','
        )
        return titles_df[[self.IID_COL, 'title']]

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'MicroLens-50k_pairs.csv')
        interactions = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
        )
        interactions = self._stringify(interactions)

        self._get_user_dict_from_interactions(interactions)

        users = interactions.sort_values(
            [self.UID_COL, 'timestamp']
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return self._extract_pos_samples(users)

        # return users
