import os

import pandas as pd

from process.base_amazon_processor import AmazonProcessor
from process.base_uict_processor import UICTProcessor

import gzip
import json

from seq_process.base_seqprocessor import BaseSeqProcessor


class AmazonSeqProcessor(BaseSeqProcessor, AmazonProcessor):
    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, f"{self.subset}.json.gz")
        interactions = self._load_data(path)

        interactions = interactions[[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL]]

        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL], format='%m %d, %Y')
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        interactions[self.RAT_COL] = interactions[self.RAT_COL].astype(int)
        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL].apply(lambda x: int(x >= 4))

        interactions = interactions.drop(columns=[self.RAT_COL])

        interactions = interactions.groupby(self.UID_COL)
        interactions = interactions.filter(lambda x: x[self.LBL_COL].nunique() == 2)

        pos_inters = interactions[interactions[self.LBL_COL] == 1]

        users = pos_inters.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return users
