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

    POS_COUNT = 2
    NEG_RATIO = 2

    NUM_TEST = 0
    NUM_FINETUNE = 100000

    REQUIRE_STRINGIFY = False

    MAX_INTERACTIONS_PER_USER = 50

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.path = os.path.join(self.data_dir, 'HotelRec.txt')

    @property
    def default_attrs(self):
        return ['hotel_name', 'hotel_location']

    @staticmethod
    def _parse_url(url):
        hotel_id = hashlib.md5(url.encode()).hexdigest()[:6]
        url = url[url.find('-Reviews-') + 9:].split('.')[0]
        hotel_name, location = url.split('-')
        hotel_name = hotel_name.replace('_', ' ')
        location = location.replace('_', ' ')
        return hotel_id, hotel_name, location

    def load_items(self) -> pd.DataFrame:
        items = []
        item_set = set()

        with open(self.path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                if index >= 2e6:
                    break

                data = json.loads(line)
                hotel_id, hotel_name, location = self._parse_url(data['hotel_url'])
                if hotel_id not in item_set:
                    items.append([hotel_id, hotel_name, location])
                    item_set.add(hotel_id)

        return pd.DataFrame(items, columns=[self.IID_COL, 'hotel_name', 'hotel_location'])

    def load_users(self) -> pd.DataFrame:
        interactions = []
        with open(self.path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                if index >= 2e6:
                    break
                data = json.loads(line)
                hotel_url, user_id, date, rating = data['hotel_url'], data['author'], data['date'], data['rating']
                hotel_id, _, _ = self._parse_url(hotel_url)
                interactions.append([user_id, hotel_id, date, int(rating)])
        interactions = pd.DataFrame(interactions, columns=[self.IID_COL, self.UID_COL, self.DAT_COL, 'rating'])

        interactions = interactions[interactions['rating'] != 3]
        interactions[self.LBL_COL] = interactions['rating'] > 3
        interactions.drop(columns=['rating'], inplace=True)

        return self._load_users(interactions)
