import os

import pandas as pd

from process.neg_processor import NegProcessor


class YelpProcessor(NegProcessor):
    UID_COL = 'user_id'
    IID_COL = 'business_id'
    HIS_COL = 'history'
    CLK_COL = 'click'

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
        interactions = interactions[['user_id', 'business_id', 'stars', 'date']]

        interactions['stars'] = interactions['stars'].astype(int)
        interactions = interactions[interactions['stars'] != 3]
        interactions['click'] = int(interactions['stars'] >= 4)
        interactions = interactions.drop(columns=['stars'])

        # group by user
        interactions = interactions.groupby('user_id')
        interactions = interactions.filter(lambda x: x['click'].nunique() > 2)
        self._interactions = interactions

        # those stars >= 4 are considered as positive and stars <= 2 are considered as negative
        pos_inters = interactions[interactions['click'] == 1]
        # format string date to datetime
        pos_inters['date'] = pd.to_datetime(pos_inters['date'])

        users = pos_inters.sort_values(
            ['user_id', 'date']
        ).groupby('user_id')['business_id'].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        users = self._generate_pos_inters_from_history(users)
        return users

    def load_interactions(self) -> pd.DataFrame:
        neg_inters = self._interactions[self._interactions['click'] == 0]
        # remove date column
        neg_inters = neg_inters.drop(columns=['date'])

        # concat with positive interactions
        return pd.concat([neg_inters, self.pos_inters], ignore_index=True)
