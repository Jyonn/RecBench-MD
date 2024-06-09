import json
import os
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from process.base_uict_processor import UICTProcessor


class GoodreadsProcessor(UICTProcessor):
    IID_COL = 'bid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    CLK_COL = 'click'
    DAT_COL = 'date'

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    REQUIRE_STRINGIFY = True

    POS_COUNT = 2

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
        items = self._stringify(items)
        return items

    @staticmethod
    def _str_to_ts(date_string):
        date_format = "%a %b %d %H:%M:%S %z %Y"
        dt = datetime.strptime(date_string, date_format)
        timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return timestamp

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'goodreads_interactions_dedup.json')
        interactions = []
        with open(path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                if index > 1e7:
                    break
                data = json.loads(line.strip())
                user_id, book_id, is_read, date = data['user_id'], data['book_id'], data['is_read'], data['date_added']
                interactions.append([user_id, book_id, is_read, date])

        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.CLK_COL, self.DAT_COL])
        interactions = self._stringify(interactions)
        interactions[self.DAT_COL] = interactions[self.DAT_COL].apply(lambda x: self._str_to_ts(x))
        interactions[self.CLK_COL] = interactions[self.CLK_COL].apply(lambda x: int(x))
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]
        return self._load_users(interactions)
