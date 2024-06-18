import pandas as pd
import torch
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, datalist: pd.DataFrame):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # return self.datalist[idx]
        values = self.datalist.iloc[idx]
        return dict(
            input_ids=torch.tensor(values['input_ids'], dtype=torch.long),
            labels=torch.tensor(values['labels'], dtype=torch.long),
            uid=torch.tensor(values['uid'], dtype=torch.long),
            iid=torch.tensor(values['iid'], dtype=torch.long),
            length=torch.tensor(values['length'], dtype=torch.long),
        )
