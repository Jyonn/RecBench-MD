import os

import pandas as pd

from process.base_ns_processor import NSProcessor
from process.base_uspe_processor import USPEProcessor
from tqdm import tqdm


class PogProcessor(NSProcessor, USPEProcessor):
    IID_COL = 'item_id'
    UID_COL = 'user_id'
    HIS_COL = 'history'
    LBL_COL = 'click'

    POS_COUNT = 1
    NEG_RATIO = 100

    NUM_TEST = 0
    NUM_FINETUNE = 100_000

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['title_en']  # 指定文本信息Title

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "pog_translated_titles.csv")
        item = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
        )
        item.columns = [self.IID_COL, "title_cn", "title_en"]
        return item

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'user_data.txt')

        self._user_dict = dict()

        users = []
        with open(path, 'r') as f:
            for index, line in tqdm(enumerate(f)):
                line = line.replace(';', ',')
                data = line.strip().split(',')
                uid, his = data[0], data[1:-1]  # 最后一个是outfit_id和item_id不一样
                his = list(filter(lambda x: x in item_set, his))
                his_, his_set = [], set()
                for item in his:
                    if item not in his_set:
                        his_.append(item)
                        his_set.add(item)
                self._user_dict[uid] = his_set
                users.append({self.UID_COL: uid, self.HIS_COL: his_})
        users = pd.DataFrame(users)

        return self._extract_pos_samples(users)
