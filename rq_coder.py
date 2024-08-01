import json

import pigmento
from oba import Obj
from pigmento import pnt

from process.base_processor import BaseProcessor
from utils.config_init import ConfigInit
from utils.function import load_processor


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data', 'rq', 'export'],
        default_args=dict(),
        makedirs=[]
    ).parse()

    processor = load_processor(configuration.data.lower())  # type: BaseProcessor
    processor.load()

    item_vocab = processor.item_vocab

    item_dict = dict()
    for item, index in item_vocab.items():
        item_dict[str(index)] = item

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
        final_dict[item_dict[item]] = _codes

    json.dump(final_dict, open(configuration.export, 'w'), indent=2)
