import pandas as pd
import torch
from pigmento import pnt
from tqdm import tqdm

from loader.dataset import Dataset
from loader.map import Map


class EmbedDataset(Dataset):
    def __len__(self):
        return len(self.datalist)

    def align(self, batch_size, ascending=False):
        self.datalist = self.datalist.sort_values(Map.UIL_COL, ascending=ascending).reset_index(drop=True)

        pnt(f'combining dataset by step-wise length alignment')
        num_batches = (len(self.datalist) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), total=num_batches):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, len(self.datalist))
            batch = self.datalist.loc[start_index:end_index - 1]
            max_len = batch[Map.UIL_COL].max()
            self.datalist.loc[start_index:end_index - 1, Map.IIP_COL] = batch[Map.IIP_COL].apply(
                lambda x: list(x)[:max_len] + [0] * (max_len - len(x)))
            self.datalist.loc[start_index:end_index - 1, Map.UIP_COL] = batch[Map.UIP_COL].apply(
                lambda x: list(x)[:max_len] + [0] * (max_len - len(x)))
