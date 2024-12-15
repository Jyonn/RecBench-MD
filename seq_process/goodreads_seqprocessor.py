import json
import os

import pandas as pd
from tqdm import tqdm

from process.goodreads_processor import GoodreadsProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor


class GoodreadsSeqProcessor(BaseSeqProcessor, GoodreadsProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    REQUIRE_STRINGIFY = True

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'goodreads_interactions_dedup.json')
        interactions = []
        with open(path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                if index > 5e7:
                    break
                data = json.loads(line.strip())
                user_id, book_id, is_read, date = data['user_id'], data['book_id'], data['is_read'], data['date_added']
                interactions.append([user_id, book_id, is_read, date])

        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.LBL_COL, self.DAT_COL])

        interactions = self._stringify(interactions)
        interactions[self.DAT_COL] = interactions[self.DAT_COL].apply(lambda x: self._str_to_ts(x))
        interactions[self.LBL_COL] = interactions[self.LBL_COL].apply(int)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        interactions = interactions.groupby(self.UID_COL)
        interactions = interactions.filter(lambda x: x[self.LBL_COL].nunique() == 2)

        pos_inters = interactions[interactions[self.LBL_COL] == 1]

        users = pos_inters.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return users
