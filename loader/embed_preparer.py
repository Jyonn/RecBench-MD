import os.path
from typing import Optional

import pandas as pd
from UniTok import Vocab
from pigmento import pnt
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.dataset import Dataset
from loader.embed_dataset import EmbedDataset
from loader.map import Map
from loader.preparer import Preparer
from model.base_model import BaseModel
from process.base_processor import BaseProcessor


class EmbedPreparer(Preparer):
    DATASET_CLASS = EmbedDataset

    def get_primary_signature(self):
        return f'{self.processor.get_name()}_{self.model.get_name()}_embed'

    def load_datalist(self):
        items = self.tokenize_items()
        line, numbers, user, item, prefix, suffix = self.model.get_special_tokens()

        datalist = []

        max_sequence_len = 0
        pnt(f'preprocessing on the {self.processor.get_name()} dataset')
        for index, data in tqdm(
                enumerate(self.processor.generate(slicer=self.conf.slicer, source='finetune', id_only=True)),
                total=len(self.processor.get_source_set(source='finetune'))
        ):
            uid, iid, history, label = data

            current_item = items[iid][:]
            item_input_ids = item + current_item

            user_length = len(user)
            idx = len(history) - 1
            while idx >= 0:
                current_len = len(items[history[idx]]) + len(numbers[len(history) - idx]) + len(line)
                if user_length + current_len <= self.model.max_len:
                    user_length += current_len
                else:
                    break
                idx -= 1

            user_input_ids = user
            for i in range(idx + 1, len(history)):
                user_input_ids += numbers[i - idx] + items[history[i]] + line

            assert user_input_ids is not None
            max_sequence_len = max(max_sequence_len, user_length)
            datalist.append({Map.IIP_COL: item_input_ids, Map.UIP_COL: user_input_ids, Map.LBL_COL: label, Map.UID_COL: uid, Map.IID_COL: iid})

        for data in datalist:
            data[Map.IIL_COL] = len(data[Map.IIP_COL])
            data[Map.IIP_COL] = data[Map.IIP_COL] + [0] * (max_sequence_len - data[Map.IIL_COL])
            data[Map.LEN_COL] = data[Map.UIL_COL] = len(data[Map.UIP_COL])
            data[Map.UIP_COL] = data[Map.UIP_COL] + [0] * (max_sequence_len - data[Map.UIL_COL])
            data[Map.UID_COL] = self.uid_vocab.append(data[Map.UID_COL])
            data[Map.IID_COL] = self.iid_vocab.append(data[Map.IID_COL])

        pnt(f'{self.processor.get_name()} dataset: max_sequence_len: {max_sequence_len}')

        return datalist
