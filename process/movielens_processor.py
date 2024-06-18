import os
import pandas as pd
from process.base_uict_processor import UICTProcessor


class MovieLensProcessor(UICTProcessor):
    IID_COL = 'movieId'
    UID_COL = 'userId'
    HIS_COL = 'his'
    LBL_COL = 'click'
    DAT_COL = 'timestamp'

    POS_COUNT = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    REQUIRE_STRINGIFY = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pos_inters = None

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'movies.csv')
        movies = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
            engine='python',
            encoding="ISO-8859-1",
        )
        movies['title'] = movies['title'].str.replace(r'[^A-Za-z0-9 ]+', '')
        return movies

    def load_users(self) -> pd.DataFrame:
        interactions = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'ratings.csv'),
            sep=',',
            engine='python'
        )

        # filter out rating = 3
        interactions = interactions[interactions['rating'] != 3]
        interactions[self.CLK_COL] = interactions['rating'] > 3
        interactions.drop(columns=['rating'], inplace=True)

        return self._load_users(interactions)
