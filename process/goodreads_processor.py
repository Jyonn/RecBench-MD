import json
import os

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

    POS_COUNT = 2

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

    # @staticmethod
    # def _str_to_ts(date_string):
    #     # 定义字符串的日期格式
    #     date_format = "%a %b %d %H:%M:%S %z %Y"
    #     # 将字符串转换为datetime对象
    #     dt = datetime.strptime(date_string, date_format)
    #     # 将datetime对象转换为timestamp（秒数）
    #     timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
    #     return timestamp

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'goodreads_interactions_dedup.json')
        interactions = []
        with open(path, 'r') as f:
            for line in tqdm(f):
                data = json.loads(line.strip())
                user_id, book_id, is_read, date = data['user_id'], data['book_id'], data['is_read'], data['date_added']
                interactions.append([user_id, book_id, is_read, date])
        # interactions = pd.read_json(path, lines=True)
        # pnt('interaction loaded')
        # interactions = interactions[['user_id', 'book_id', 'is_read', 'date_added']]
        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.CLK_COL, self.DAT_COL])
        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL])
        interactions[self.CLK_COL] = interactions[self.CLK_COL].apply(lambda x: int(x))
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        return self._load_users(interactions)
