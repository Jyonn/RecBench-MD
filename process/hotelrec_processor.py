import hashlib
import json
import os.path

import pandas as pd
from tqdm import tqdm

from process.base_uict_processor import UICTProcessor


class HotelRecProcessor(UICTProcessor):
    UID_COL = 'uid'
    IID_COL = 'hid'
    LBL_COL = 'click'
    HIS_COL = 'history'
    DAT_COL = 'date'
    RAT_COL = 'rating'

    POS_COUNT = 2
    NEG_RATIO = 2

    NUM_TEST = 20_000
    NUM_FINETUNE = 0

    REQUIRE_STRINGIFY = False

    MAX_INTERACTIONS_PER_USER = 50

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def default_attrs(self):
        return ['hotel_name', 'hotel_location']

    @staticmethod
    def _parse_url(url):
        hotel_id = hashlib.md5(url.encode()).hexdigest()[:6]
        url = url[url.find('-Reviews-') + 9:].split('.')[0]
        if '-' not in url:
            return None, None, None
        hotel_name, location = url.split('-')
        hotel_name = hotel_name.replace('_', ' ')
        location = location.replace('_', ' ')
        return hotel_id, hotel_name, location

    def load_items(self) -> pd.DataFrame:
        items = []
        item_set = set()

        path = os.path.join(self.data_dir, 'HotelRec.txt')
        with open(path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                if index >= 2e6:
                    break

                data = json.loads(line)
                hotel_id, hotel_name, location = self._parse_url(data['hotel_url'])
                if not hotel_id:
                    continue
                if hotel_id not in item_set:
                    items.append([hotel_id, hotel_name, location])
                    item_set.add(hotel_id)

        items = pd.DataFrame(items, columns=[self.IID_COL, 'hotel_name', 'hotel_location'])
        items[self.IID_COL] = items[self.IID_COL].astype(str)
        return items

    def load_users(self) -> pd.DataFrame:
        interactions = []
        path = os.path.join(self.data_dir, 'HotelRec.txt')
        with open(path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                if index >= 2e6:
                    break
                data = json.loads(line)
                hotel_url, user_id, date, rating = data['hotel_url'], data['author'], data['date'], data['rating']
                hotel_id, _, _ = self._parse_url(hotel_url)
                interactions.append([user_id, hotel_id, date, int(rating)])

        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.DAT_COL, self.RAT_COL])

        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL] > 3
        interactions[self.LBL_COL] = interactions[self.LBL_COL].apply(int)
        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL])
        interactions.drop(columns=[self.RAT_COL], inplace=True)

        return self._load_users(interactions)
