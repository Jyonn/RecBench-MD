import os.path
from typing import Optional

import pandas as pd
from UniTok import Vocab
from pigmento import pnt
from torch.utils.data import DataLoader

from loader.dataset import Dataset
from model.base_model import BaseModel
from process.base_processor import BaseProcessor
from utils.tqdm_printer import TqdmPrinter


class Preparer:
    def __init__(self, processor: BaseProcessor, model: BaseModel, conf):
        self.processor = processor
        self.model = model
        self.conf = conf

        self.store_dir = os.path.join(
            'prepare',
            f'{self.processor.get_name()}_{self.model.get_name()}, '
            f'ratio_{self.conf.valid_ratio}'
        )
        os.makedirs(self.store_dir, exist_ok=True)
        self.iid_vocab = Vocab(name='iid')
        self.uid_vocab = Vocab(name='uid')

        self.train_datapath = os.path.join(self.store_dir, 'train.parquet')
        self.valid_datapath = os.path.join(self.store_dir, 'valid.parquet')

        self.has_generated = os.path.exists(self.train_datapath) and os.path.exists(self.valid_datapath)

    def tokenize_items(self, source='finetune', item_attrs=None):
        item_set = self.processor.get_item_subset(source, slicer=self.conf.slicer)
        item_attrs = item_attrs or self.processor.default_attrs

        item_dict = dict()
        for iid in item_set:
            item_str = self.processor.organize_item(iid, item_attrs)
            item_ids = self.model.generate_simple_input_ids(item_str)
            item_dict[iid] = item_ids
        return item_dict

    def load_datalist(self):
        items = self.tokenize_items()
        line, numbers, user, item, prefix, suffix = self.model.get_special_tokens()

        datalist = []

        TqdmPrinter.activate()
        max_sequence_len = 0
        for index, data in enumerate(self.processor.generate(slicer=self.conf.slicer, source='finetune', id_only=True)):
            pnt(f'preprocessing on the {self.processor.get_name()} dataset', current=index + 1, count=len(self.processor.finetune_set))
            uid, iid, history, click = data

            current_item = items[iid][:]
            init_length = len(prefix) + len(user) + len(suffix)
            input_ids: Optional[list] = None
            for _ in range(5):
                current_length = init_length + len(current_item)

                index = len(history) - 1
                while index >= 0:
                    current_len = len(items[history[index]]) + len(numbers[len(history) - index]) + len(line)
                    if current_length + current_len <= self.model.max_len:
                        current_length += current_len
                    else:
                        break
                    index -= 1

                if index == len(history) - 1:
                    current_item = current_item[:len(current_item) // 2]
                    continue

                input_ids = prefix + user
                for i in range(max(index, 0), len(history)):
                    input_ids += numbers[len(history) - i] + items[history[i]] + line
                input_ids += item + current_item + suffix
                break

            assert input_ids is not None, f'failed to get input_ids for {index} ({uid}, {iid})'
            max_sequence_len = max(max_sequence_len, len(input_ids))
            datalist.append({'input_ids': input_ids, 'labels': click, 'uid': uid, 'iid': iid})
        TqdmPrinter.deactivate()

        for data in datalist:
            data['length'] = len(data['input_ids'])
            data['input_ids'] = data['input_ids'] + [0] * (max_sequence_len - data['length'])
            data['uid'] = self.uid_vocab.append(data['uid'])
            data['iid'] = self.iid_vocab.append(data['iid'])

        return datalist

    def split_datalist(self, datalist):
        valid_user_set = self.processor.load_valid_user_set(self.conf.valid_ratio)

        train_datalist = []
        valid_datalist = []
        for data in datalist:
            if data[self.processor.UID_COL] in valid_user_set:
                valid_datalist.append(data)
            else:
                train_datalist.append(data)

        return train_datalist, valid_datalist

    def load_or_generate(self):
        if self.has_generated:
            pnt(f'loading prepared finetuning data on {self.processor.get_name()} dataset')
            train_datalist = pd.read_parquet(self.train_datapath)
            valid_datalist = pd.read_parquet(self.valid_datapath)

            self.iid_vocab.load(self.store_dir)
            self.uid_vocab.load(self.store_dir)
        else:
            datalist = self.load_datalist()
            train_datalist, valid_datalist = self.split_datalist(datalist)

            train_datalist = pd.DataFrame(train_datalist)
            valid_datalist = pd.DataFrame(valid_datalist)

            self.iid_vocab.save(self.store_dir)
            self.uid_vocab.save(self.store_dir)

            train_datalist.to_parquet(self.train_datapath)
            valid_datalist.to_parquet(self.valid_datapath)

        train_dataset = Dataset(train_datalist)
        valid_dataset = Dataset(valid_datalist)

        train_dataloader = DataLoader(train_dataset, batch_size=self.conf.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.conf.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader
