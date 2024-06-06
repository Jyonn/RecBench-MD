import os
import pandas as pd
from process.base_processor import BaseProcessor


class MovieLensProcessor(BaseProcessor):
    IID_COL = 'mid'
    UID_COL = 'uid'
    HIS_COL = 'his'
    CLK_COL = 'click'

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
            engine='python',encoding="ISO-8859-1"
        )
        # movie title may exist special characters, so we need to remove them
        movies['title'] = movies['title'].str.replace(r'[^A-Za-z0-9 ]+', '')
        return movies

    def load_users(self) -> pd.DataFrame:
        ratings = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'ratings.dat'),
            sep='::',
            header=None,
            names=['uid', 'mid', 'rating', 'ts'],
            engine='python'
        )

        ratings = ratings[ratings['rating'] > 3]
        users = ratings.sort_values(by=['uid', 'ts']).groupby('uid')['mid'].apply(list).reset_index()

        users.columns = [self.UID_COL, self.HIS_COL]

        self.pos_inters = users.copy()
        self.pos_inters[self.IID_COL] = self.pos_inters[self.HIS_COL].apply(lambda x: x[-1])
        self.pos_inters.drop(columns=[self.HIS_COL], inplace=True)
        self.pos_inters[self.CLK_COL] = 1

        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: x[:-1])
        return users

    def load_interactions(self) -> pd.DataFrame:
        interactions = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'ratings.dat'),
            sep='::',
            header=None,
            names=['uid', 'mid', 'rating', 'ts'],
            engine='python'
        )

        interactions = interactions[interactions['rating'] < 3]
        interactions.drop(columns=['rating', 'ts'], inplace=True)
        interactions[self.CLK_COL] = 0

        # merge with positive interactions
        interactions = pd.concat([interactions, self.pos_inters], ignore_index=True)
        return interactions
