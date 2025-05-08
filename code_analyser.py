import json

import pigmento

from pigmento import pnt

from seq_process.base_seqprocessor import BaseSeqProcessor
from utils.config_init import ConfigInit
from utils.function import load_seq_processor

if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['code_path'],
        default_args=dict(
            data=None
        ),
        makedirs=[]
    ).parse()

    indices = json.load(open(configuration.code_path))

    code_set = set()
    for _, value in indices.items():
        code_set.add('-'.join(map(str, value)))

    pnt(f'code set size: {len(code_set)}')
    pnt(f'indices size: {len(indices)}')
    pnt(f'collapse rate: {1 - len(code_set) / len(indices)}')

    if configuration.data is None:
        exit(0)

    processor: BaseSeqProcessor = load_seq_processor(configuration.data)
    processor.load()

    code_set = dict()
    code_dict = dict()
    for item_id in processor.items[processor.IID_COL]:
        value = indices[item_id]
        current_code = '-'.join(map(str, value))
        if current_code not in code_set:
            code_set[current_code] = 1
            code_dict[current_code] = []
        code_set[current_code] += 1
        code_dict[current_code].append(item_id)

    o2i = dict(zip(processor.items[processor.IID_COL], range(len(processor.items))))

    print(f'code set size: {len(code_set)}')
    print(f'item size: {len(processor.items)}')
    print(f'collapse rate: {1 - len(code_set) / len(processor.items)}')
    print(f'max code count: {max(code_set.values())}')

    for code in code_set:
        if code_set[code] > 200:
            print(code, code_set[code])
            for item_id in code_dict[code]:
                index = o2i[item_id]
                for attr in processor.default_attrs:
                    print(processor.items.iloc[index][attr])
