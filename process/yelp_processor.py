import os

import pandas as pd

from process.base_uict_processor import UICTProcessor


class YelpProcessor(UICTProcessor):
    UID_COL = 'user_id'
    IID_COL = 'business_id'
    HIS_COL = 'history'
    CLK_COL = 'click'
    DAT_COL = 'date'

    POS_COUNT = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

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
        interactions = interactions[[self.UID_COL, self.IID_COL, 'stars', self.DAT_COL]]

        interactions['stars'] = interactions['stars'].astype(int)
        interactions = interactions[interactions['stars'] != 3]
        interactions['click'] = interactions['stars'].apply(lambda x: int(x > 3))
        interactions = interactions.drop(columns=['stars'])

        interactions['date'] = pd.to_datetime(interactions['date'])

        return self._load_users(interactions)
