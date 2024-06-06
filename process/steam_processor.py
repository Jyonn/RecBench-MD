import os

import pandas as pd

from process.neg_processor import NegProcessor


class SteamProcessor(NegProcessor):
    IID_COL = 'app_id'
    UID_COL = 'user_id'
    HIS_COL = 'history'
    CLK_COL = 'click'

    NUM_TEST = 20000  # test number #指定测试数量 前10个是固定的，后5个为0
    NUM_FINETUNE = 100000  # 固定的数量

    @property
    def default_attrs(self):
        return ['title']  # 指定文本信息Title

    def load_items(self) -> pd.DataFrame:
        apps = pd.read_csv(os.path.join(self.data_dir, "App_ID_Info.txt"), header=None)
        apps = apps.iloc[:, [0, 1]]

        apps.columns = ['app_id', 'title']
        apps['title'] = apps['title'].apply(lambda x: x.encode('latin1').decode('utf-8', 'ignore'))

        return apps

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'train_game.txt')
        users = []
        self.user_dict = dict()
        pos_inters = []

        with open(path, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                uid, his = data[0], data[1:]
                his = list(filter(lambda x: x in item_set, his))
                if len(his) < 2:
                    continue
                self.user_dict[uid] = set(his)
                users.append({'user_id': uid, 'history': his[:-2]})
                pos_inters.append({'user_id': uid, 'app_id': his[-2], 'click': 1})
                pos_inters.append({'user_id': uid, 'app_id': his[-1], 'click': 1})

        self.pos_inters = pd.DataFrame(pos_inters)
        return pd.DataFrame(users)
