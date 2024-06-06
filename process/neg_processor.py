import abc
import random

import pandas as pd

from process.base_processor import BaseProcessor


class NegProcessor(BaseProcessor, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.user_dict = None
        self.pos_inters = None

        self.neg_ratio = 2
        self.pos_count = 2

    def load_interactions(self) -> pd.DataFrame:
        assert self.user_dict is not None and self.pos_inters is not None, 'User dict and positive interactions must be generated first'

        items = self.items[self.IID_COL].unique().tolist()
        num_items = len(items)
        interactions = []

        for user_id in self.user_dict:
            user_interactions = self.user_dict[user_id]
            for _ in range(self.neg_ratio):
                neg_item = items[random.randint(0, num_items - 1)]
                while neg_item in user_interactions:
                    neg_item = items[random.randint(0, num_items - 1)]
                interactions.append({self.UID_COL: user_id, self.IID_COL: neg_item, self.CLK_COL: 0})
        interactions = pd.DataFrame(interactions)
        interactions = pd.concat([interactions, self.pos_inters], ignore_index=True)
        return interactions

    def _generate_pos_inters_from_history(self, users):
        users = users[users[self.HIS_COL].apply(len) > self.pos_count]

        pos_inters = []
        for index, row in users.iterrows():
            for i in range(self.pos_count):
                pos_inters.append({
                    self.UID_COL: row[self.UID_COL],
                    self.IID_COL: row[self.HIS_COL][-(i + 1)],
                    self.CLK_COL: 1
                })
        self.pos_inters = pd.DataFrame(pos_inters)

        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: x[:-self.pos_count])
        return users
