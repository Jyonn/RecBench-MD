import os

import pandas as pd

from process.microlens_processor import MicroLensProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor


class MicroLensSeqProcessor(BaseSeqProcessor, MicroLensProcessor):
    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    REQUIRE_STRINGIFY = True

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'MicroLens-50k_pairs.csv')
        interactions = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
        )
        interactions = self._stringify(interactions)

        users = interactions.sort_values(
            [self.UID_COL, 'timestamp']
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return users
