import json
import os

import pandas as pd
from tqdm import tqdm

from process.goodreads_processor import GoodreadsProcessor
from process.hm_processor import HMProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor


class HMSeqProcessor(BaseSeqProcessor, HMProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    REQUIRE_STRINGIFY = True

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())
        path = os.path.join(self.data_dir, "transactions_train.csv")
        interactions = []
        with open(path, 'r') as f:
            f.readline()
            for index, line in tqdm(enumerate(f)):
                if index > 1e7:
                    break
                interactions.append(line.strip().split(',')[:3])
        interactions = pd.DataFrame(interactions, columns=[self.DAT_COL, self.UID_COL, self.IID_COL])
        # filter out items not in item set
        interactions = self._stringify(interactions)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        users = interactions.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return users
