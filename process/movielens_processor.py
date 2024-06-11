import os
import pandas as pd
from process.base_uict_processor import UICTProcessor


class MovieLensProcessor(UICTProcessor):
    IID_COL = 'mid'
    UID_COL = 'uid'
    HIS_COL = 'his'
    CLK_COL = 'click'
    DAT_COL = 'ts'

    POS_COUNT = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pos_inters = None

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'movies.dat')
        movies = pd.read_csv(
            filepath_or_buffer=path,
            sep='::',
            header=None,
            names=['mid', 'title', 'genres'],
            engine='python',
            encoding="ISO-8859-1",
        )
        # movie title may exist special characters, so we need to remove them
        movies['title'] = movies['title'].str.replace(r'[^A-Za-z0-9 ]+', '')
        return movies

    def load_users(self) -> pd.DataFrame:
        interactions = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'ratings.dat'),
            sep='::',
            header=None,
            names=[self.UID_COL, self.IID_COL, 'rating', self.DAT_COL],
            engine='python'
        )

        # filter out rating = 3
        interactions = interactions[interactions['rating'] != 3]
        interactions[self.CLK_COL] = interactions['rating'] > 3
        interactions.drop(columns=['rating'], inplace=True)

        return self._load_users(interactions)
