import torch

from loader.dataset import Dataset
from loader.dense_code_map import DenseCodeMap as Map


class DenseCodeDataset(Dataset):
    def __getitem__(self, idx):
        values = self.datalist.iloc[idx]
        return {
            Map.IPT_COL: torch.tensor(values[Map.IPT_COL], dtype=torch.long),
            Map.VOC_COL: torch.tensor(values[Map.VOC_COL], dtype=torch.long),
            Map.LBL_COL: torch.tensor(values[Map.LBL_COL], dtype=torch.long),
            Map.UID_COL: torch.tensor(values[Map.UID_COL], dtype=torch.long),
            Map.IID_COL: torch.tensor(values[Map.IID_COL], dtype=torch.long),
            Map.LEN_COL: torch.tensor(values[Map.LEN_COL], dtype=torch.long),
        }
