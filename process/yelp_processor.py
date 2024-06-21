import os

import pandas as pd

from process.base_uict_processor import UICTProcessor


class YelpProcessor(UICTProcessor):
    UID_COL = 'user_id'
    IID_COL = 'business_id'
    HIS_COL = 'history'
    LBL_COL = 'click'
    DAT_COL = 'date'
    RAT_COL = 'stars'

    POS_COUNT = 2

    NUM_TEST = 20_000
    NUM_FINETUNE = 0

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['name']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'yelp_academic_dataset_business.json')
        items = pd.read_json(path, lines=True)
        items = items[['business_id', 'name', 'address', 'city', 'state']]
        return items

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'yelp_academic_dataset_review.json')
        interactions = pd.read_json(path, lines=True)
        interactions = interactions[[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL]]

        interactions[self.RAT_COL] = interactions[self.RAT_COL].astype(int)
        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL].apply(lambda x: int(x > 3))
        interactions = interactions.drop(columns=[self.RAT_COL])

        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL])

        return self._load_users(interactions)
