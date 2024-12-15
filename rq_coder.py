import json

import pigmento
from oba import Obj
from pigmento import pnt

from process.base_processor import BaseProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor
from utils.config_init import ConfigInit
from utils.function import load_processor, load_seq_processor

if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data', 'rq', 'export'],
        default_args=dict(
            seq=False,
        ),
        makedirs=[]
    ).parse()

    if configuration.seq:
        processor = load_seq_processor(configuration.data.lower())  # type: BaseSeqProcessor
    else:
        processor = load_processor(configuration.data.lower())  # type: BaseProcessor
    processor.load()

    item_vocab = processor.item_vocab
    item_dict = dict()
    max_index, min_index = 0, 1000000
    for item, index in item_vocab.items():
        item_dict[str(index)] = item
        max_index = max(max_index, index)
        min_index = min(min_index, index)
    print(f'item vocab: {len(item_vocab)} items, max index: {max_index}, min index: {min_index}')

    item_dict = dict(zip(range(len(processor.items)), processor.items[processor.IID_COL]))
    print(f'item dict: {len(item_dict)}, items: {len(processor.items)}')

    print(processor.items)

    item_code = Obj.raw(configuration.rq)
    if isinstance(item_code, str):
        item_code = json.load(open(item_code))
    final_dict = dict()

    for item, codes in item_code.items():
        _codes = []
        for code in codes:
            code = code.split('_')[1]
            code = int(code[:-1])
            _codes.append(code)
        final_dict[item_dict[int(item)]] = _codes

    json.dump(final_dict, open(configuration.export, 'w'), indent=2)
