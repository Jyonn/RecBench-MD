import os

import pandas as pd

from process.base_uict_processor import UICTProcessor


class AmazonProcessor(UICTProcessor):
    UID_COL = 'reviewerID'
    IID_COL = 'asin'
    #HIS_COL = 'history'
    #CLK_COL = 'click'
    DAT_COL = 'reviewTime'

    POS_COUNT = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    @property
    def default_attrs(self):
        return ['reviewText']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._interactions = None

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "reviews_Electronics.csv")
        items = pd.read_json((path,lines=True)
        items = items[['asin', 'reviewText']]
        #filter special symbol
        items['title'] = items['title'].str.replace(r'&#[0-9]+;', '', regex=True)
        items['title'] = items['title'].str.replace(r'&[a-zA-Z]+;', '', regex=True)
        items['title'] = items['title'].str.replace(r'[^\w\s]', '', regex=True)
        return items

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())                    
        columns = [self.UID_COL, self.IID_COL, 'overall', self.DAT_COL]
        
        interactions = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'reviews_Electronics_5.json.gz'),
            sep='::',
            header=None,
            names=columns,
            engine='python'
        )
        
        interactions[self.DAT_COL]= pd.to_datetime(interactions[self.DAT_COL], unit='s')
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]
                             
                             
        interactions['overall'] = interactions['overall'].astype(int)
        interactions = interactions[interactions['overall'] != 3]
        #interactions['click'] = int(interactions['overall'] >= 4)
        interactions['click'] = interactions['overall'].apply(lambda x: int(x >= 4))
        
        
        interactions = interactions.drop(columns=['overall'])

        return self._load_users(interactions)
