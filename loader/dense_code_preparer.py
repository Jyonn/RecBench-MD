import os.path
from typing import cast

import numpy as np
from UniTok import Vocab
from pigmento import pnt

from loader.dense_code_map import DenseCodeMap as Map
from loader.preparer import Preparer
from loader.token_vocab import TV


class DenseCodePreparer(Preparer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.item_embeds = cast(dict, np.load(self.conf.item_embeds).item())

        assert os.path.exists(self.iid_vocab.get_store_path(self.store_dir)) and \
                os.path.exists(self.uid_vocab.get_store_path(self.store_dir)), \
            'iid_vocab and uid_vocab should be prepared before loading DenseCodePreparer'
        self.iid_vocab.load(self.store_dir)
        self.uid_vocab.load(self.store_dir)
        self.cod_vocab = Vocab(name=Map.COD_COL)

    def tokenize_items(self, source='finetune', item_attrs=None):
        item_set = self.processor.get_item_subset(source, slicer=self.conf.slicer)

        item_dict = dict()
        for iid in item_set:
            item_embed = self.item_embeds[iid]  # type: np.ndarray
            num_embeds = len(item_embed)
            item_dict[iid] = []
            for i in range(num_embeds):
                item_dict[iid].append(self.cod_vocab.append(f'{iid}_{i}'))
        return item_dict

    def load_datalist(self):
        items = self.tokenize_items()
        line, numbers, user, item, prefix, suffix = self.model.get_special_tokens()

        datalist = []

        max_sequence_len = 0
        for index, data in enumerate(self.processor.generate(slicer=self.conf.slicer, source='finetune', id_only=True)):
            pnt(f'preprocessing on the {self.processor.get_name()} dataset', current=index + 1, count=len(self.processor.finetune_set))
            uid, iid, history, label = data

            current_item = items[iid][:]

            input_ids = prefix + user
            vocab_ids = [TV.LLM] * len(input_ids)

            for i in range(len(history)):
                input_ids += numbers[i + 1] + items[history[i]] + line
                vocab_ids += [TV.LLM] * len(numbers[i + 1]) + [TV.COD] * len(items[history[i]]) + [TV.LLM] * len(line)
            input_ids += item + current_item + suffix
            vocab_ids += [TV.LLM] * len(item) + [TV.COD] * len(current_item) + [TV.LLM] * len(suffix)

            max_sequence_len = max(max_sequence_len, len(input_ids))
            datalist.append({
                Map.IPT_COL: input_ids,
                Map.VOC_COL: vocab_ids,
                Map.LBL_COL: label,
                Map.UID_COL: uid,
                Map.IID_COL: iid
            })

        for data in datalist:
            data[Map.LEN_COL] = len(data[Map.IPT_COL])
            data[Map.IPT_COL] = data[Map.IPT_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.VOC_COL] = data[Map.VOC_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.UID_COL] = self.uid_vocab.append(data[Map.UID_COL])
            data[Map.IID_COL] = self.iid_vocab.append(data[Map.IID_COL])

        pnt(f'{self.processor.get_name()} dataset: max_sequence_len: {max_sequence_len}')

        return datalist
