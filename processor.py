from pigmento import pnt

from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_processor


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data'],
        default_args=dict(slicer=-20, use_cache=True),
        makedirs=[]
    ).parse()

    data = configuration.data.lower()
    data_dir = get_data_dir(data)
    processor = load_processor(data, use_cache=configuration.use_cache, data_dir=data_dir)
    processor.load()

    count = 0
    for uid, iid, history, candidate, click in processor.iterate(slicer=configuration.slicer):
        # print(uid, iid, history, candidate, click)
        print(f'User: {uid}, Item: {iid}, History, Click: {click}')
        print(f'History:')
        for i, h in enumerate(history):
            print(f'\t{i:2d}: {h}')
        print(f'Candidate: {candidate}')

        count += 1
        if count > 2:
            break

    test_set_user = len(processor.test_set[processor.UID_COL].unique())
    fine_tune_set_user = len(processor.finetune_set[processor.UID_COL].unique())

    pnt(f'Test set: {len(processor.test_set)} with {test_set_user} users')
    pnt(f'Finetune set: {len(processor.finetune_set)} with {fine_tune_set_user} users')
