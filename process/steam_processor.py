import os

import pandas as pd

from process.base_ns_processor import NSProcessor
from process.base_uspe_processor import USPEProcessor


class SteamProcessor(NSProcessor, USPEProcessor):
    IID_COL = 'app_id'
    UID_COL = 'user_id'
    HIS_COL = 'history'
    CLK_COL = 'click'

    POS_COUNT = 2
    NEG_RATIO = 2

    NUM_TEST = 20000
    NUM_FINETUNE = 100000

    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['title']  # 指定文本信息Title

    def load_items(self) -> pd.DataFrame:
        apps = pd.read_csv(os.path.join(self.data_dir, "App_ID_Info.txt"), header=None)
        apps = apps.iloc[:, [0, 1]]

        apps.columns = [self.IID_COL, 'title']
        apps['title'] = apps['title'].apply(lambda x: x.encode('latin1').decode('utf-8', 'ignore'))
        return self._stringify(apps)

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'train_game.txt')
        self._user_dict = dict()

        users = []

        with open(path, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                uid, his = data[0], data[1:]
                his = list(filter(lambda x: x in item_set, his))
                self._user_dict[uid] = set(his)
                users.append({self.UID_COL: uid, self.HIS_COL: his})

        users = pd.DataFrame(users)
        return self._extract_pos_samples(users)
