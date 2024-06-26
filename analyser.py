import pigmento
from pigmento import pnt

from process.base_processor import BaseProcessor
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_processor


pigmento.add_time_prefix()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data'],
        default_args=dict(),
        makedirs=[]
    ).parse()

    data = configuration.data.lower()
    data_dir = get_data_dir(data)
    processor = load_processor(data, data_dir=data_dir)  # type: BaseProcessor
    processor.load()

    names = ['test', 'finetune']
    for name in names:
        subset = processor.get_source_set(name)
        if subset is None:
            continue

        pnt(f'{name} set: {len(subset)}')

        item_num = subset[processor.IID_COL].nunique()
        user_num = subset[processor.UID_COL].nunique()

        pnt(f'Item: {item_num}, User: {user_num}')
