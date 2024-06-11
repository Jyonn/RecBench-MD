import os

import pandas as pd

from process.base_ns_processor import NSProcessor
from process.base_uspe_processor import USPEProcessor
from tqdm import tqdm

class HMProcessor(NSProcessor, USPEProcessor):
    IID_COL = 'article_id'
    UID_COL = 'customer_id'
    HIS_COL = 'history'
    CLK_COL = 'click'

    POS_COUNT = 2
    NEG_RATIO = 2
    

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    @property
    def default_attrs(self):
        return ['detail_desc']  # 指定文本信息Title

    def load_items(self) -> pd.DataFrame:
        article = pd.read_csv(os.path.join(self.data_dir, "articles.csv"))

        article=article[["article_id","detail_desc"]]
        article['article_id']=article['article_id'].astype(str)
        return article

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())
        interactions = pd.read_csv(os.path.join(self.data_dir, "transactions_train.csv"))
        interactions = interactions[[self.UID_COL,self.IID_COL,'t_dat']]
        
        interactions = self._stringify(interactions)
        
        
        interactions = interactions[interactions['article_id'].isin(item_set)]
        
        self._get_user_dict_from_interactions(interactions)

        users = interactions.sort_values(
            [self.UID_COL, 't_dat']
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        self._extract_pos_samples(users)
        # import pdb
        # pdb.set_trace()
        return users
