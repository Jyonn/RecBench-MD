from pigmento import pnt
from tqdm import tqdm

from loader.discrete_code_dataset import DiscreteCodeDataset
from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.code_map import SeqCodeMap as Map
from loader.token_vocab import TV
from seq_model.base_seqmodel import BaseSeqModel


class SeqPreparer(DiscreteCodePreparer):
    DATASET_CLASS = DiscreteCodeDataset
    model: BaseSeqModel

    def _process(self, source='finetune'):
        items = self.tokenize_items()

        line, numbers, user, item, prefix = self.model.get_special_tokens()

        datalist = []

        max_sequence_len = 0
        pnt(f'preprocessing on the {self.processor.get_name()} dataset')

        for index, data in tqdm(
                enumerate(self.processor.generate(slicer=self.conf.slicer, source=source, id_only=True)),
                total=len(self.processor.get_source_set(source=source))
        ):
            uid, history = data
            history, current_item = history[:-1], history[-1]
            current_item = items[current_item]

            input_ids = prefix + user
            vocab_ids = [TV.LLM] * len(input_ids)
            beam_start = len(input_ids)
            beam_length = len(current_item)

            for i in range(len(history)):
                input_ids += numbers[i + 1] + items[history[i]] + line
                vocab_ids += [TV.LLM] * len(numbers[i + 1]) + [TV.COD] * len(items[history[i]]) + [TV.LLM] * len(line)
                beam_start += len(numbers[i + 1]) + len(items[history[i]]) + len(line)
            input_ids += item + current_item
            vocab_ids += [TV.LLM] * len(item) + [TV.COD] * len(current_item)
            beam_start += len(item)

            max_sequence_len = max(max_sequence_len, len(input_ids))
            datalist.append({
                Map.IPT_COL: input_ids,
                Map.VOC_COL: vocab_ids,
                Map.SOB_COL: beam_start,
                Map.LOB_COL: beam_length,
                Map.UID_COL: uid,
            })

        for data in datalist:
            data[Map.LEN_COL] = len(data[Map.IPT_COL])
            data[Map.IPT_COL] = data[Map.IPT_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.VOC_COL] = data[Map.VOC_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.UID_COL] = self.uid_vocab.append(data[Map.UID_COL])

        pnt(f'{self.processor.get_name()} dataset: max_sequence_len: {max_sequence_len}')

        return datalist
