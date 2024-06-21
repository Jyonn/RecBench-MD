import pandas as pd
import torch
from torch.utils.data import Dataset as BaseDataset

from loader.map import Map


class Dataset(BaseDataset):
    def __init__(self, datalist: pd.DataFrame):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def align(self):
        max_len = self.datalist[Map.LEN_COl].max()
        self.datalist[Map.IPT_COl] = self.datalist[Map.IPT_COl].apply(lambda x: x + [0] * (max_len - len(x)))

    def __getitem__(self, idx):
        # return self.datalist[idx]
        values = self.datalist.iloc[idx]
        # return {k: torch.tensor(values[k], dtype=torch.long) for k in values}
        return {
            Map.IPT_COl: torch.tensor(values[Map.IPT_COl], dtype=torch.long),
            Map.LBL_COl: torch.tensor(values[Map.LBL_COl], dtype=torch.long),
            Map.UID_COL: torch.tensor(values[Map.UID_COL], dtype=torch.long),
            Map.IID_COL: torch.tensor(values[Map.IID_COL], dtype=torch.long),
            Map.LEN_COl: torch.tensor(values[Map.LEN_COl], dtype=torch.long),
        }
