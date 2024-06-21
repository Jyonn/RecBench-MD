import os.path

import pandas as pd

from process.base_ns_processor import NSProcessor
from process.base_uspe_processor import USPEProcessor


class LastFMProcessor(USPEProcessor, NSProcessor):
    UID_COL = 'uid'
    IID_COL = 'tid'
    LBL_COL = 'click'
    HIS_COL = 'history'
    DAT_COL = 'time'

    POS_COUNT = 10
    NEG_RATIO = 100

    NUM_TEST = 0
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = False

    MAX_INTERACTIONS_PER_USER = 100

    @property
    def default_attrs(self):
        return ['track_name', 'artist_name']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'userid-timestamp-artid-artname-traid-traname.tsv')
        items = []

        with open(path, 'r') as f:
            for line in f:
                data = line.strip().split('\t')
                user_id, date, artist_id, artist_name, track_id, track_name = data
                if not track_id.strip():
                    continue
                items.append([track_id, track_name, artist_name])

        return pd.DataFrame(items, columns=[self.IID_COL, 'track_name', 'artist_name'])

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'userid-timestamp-artid-artname-traid-traname.tsv')

        interactions = pd.read_csv(
            filepath_or_buffer=path,
            sep='\t',
            header=None,
            names=[self.UID_COL, self.DAT_COL, 'artist_id', 'artist_name', self.IID_COL, 'track_name'],
            usecols=[self.UID_COL, self.DAT_COL, self.IID_COL],
        )

        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL])
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        self._get_user_dict_from_interactions(interactions)

        users = interactions.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return self._extract_pos_samples(users)
