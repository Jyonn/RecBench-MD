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
        required_args=['code_path', 'data'],
        default_args=dict(
        ),
        makedirs=[]
    ).parse()

    indices = json.load(open(configuration.code_path))

    processor: BaseSeqProcessor = load_seq_processor(configuration.data)
    processor.load()

    code_dict = dict()
    for item_id in processor.items[processor.IID_COL]:
        value = indices[item_id]
        current_code = '-'.join(map(str, value))
        if current_code not in code_dict:
            code_dict[current_code] = []
        code_dict[current_code].append(item_id)

    max_len = 0
    for code in code_dict:
        max_len = max(max_len, len(code_dict[code]))
    if max_len == 1:
        print('No code collapse')
        exit(0)

    for code in code_dict:
        for index, item_id in enumerate(code_dict[code]):
            indices[item_id].append(index)

    json.dump(indices, open(configuration.code_path.replace('.code', '.helper.code'), 'w'), indent=4)
