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
    DAT_COL = 't_dat'

    POS_COUNT = 2
    NEG_RATIO = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 10000

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['detail_desc']  # 指定文本信息Title

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "articles.csv")
        article = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
            dtype={self.IID_COL: str}
        )

        article = article[[self.IID_COL, "detail_desc"]]
        article[self.IID_COL] = article[self.IID_COL].astype(str)
        return article

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

        interactions = self._stringify(interactions)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]
        self._get_user_dict_from_interactions(interactions)

        users = interactions.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]

        return self._extract_pos_samples(users)
