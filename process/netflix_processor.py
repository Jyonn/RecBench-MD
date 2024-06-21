import os
import pandas as pd
from process.base_uict_processor import UICTProcessor


class NetflixProcessor(UICTProcessor):
    IID_COL = 'mid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    LBL_COL = 'click'
    DAT_COL = 'date'
    RAT_COL = 'rating'

    POS_COUNT = 2

    NUM_TEST = 0
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['title', 'year']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'movie_titles.csv')
        movies = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
            header=None,
            names=[self.IID_COL, 'year', 'title'],
            encoding="ISO-8859-1",
            on_bad_lines='skip',
        )
        movies[self.IID_COL] = movies[self.IID_COL].astype(str)
        return movies

    def load_users(self) -> pd.DataFrame:
        interactions = []

        current_movie_id = None
        for i in range(1, 4):
            path = os.path.join(self.data_dir, f'combined_data_{i}.txt')
            with open(path, 'r') as f:
                for line in f:
                    if ':' in line:
                        current_movie_id = line.split(':')[0]
                        continue
                    uid, rating, date = line.strip().split(',')
                    interactions.append([uid, current_movie_id, int(rating), date])

        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL])
        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL] > 3
        interactions[self.LBL_COL] = interactions[self.LBL_COL].apply(int)
        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL])
        interactions.drop(columns=[self.RAT_COL], inplace=True)

        return self._load_users(interactions)
