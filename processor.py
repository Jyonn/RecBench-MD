import pigmento
from pigmento import pnt
from tqdm import tqdm

from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_processor


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data'],
        default_args=dict(
            slicer=-20,
            source='original',
        ),
        makedirs=[]
    ).parse()

    data = configuration.data.lower()
    data_dir = get_data_dir(data)
    processor = load_processor(data, data_dir=data_dir)
    processor.load()

    # interactions.filter(lambda x: x[self.CLK_COL].nunique() == 2)
    groups = processor.interactions.groupby(processor.UID_COL)
    groups = groups.filter(lambda x: x[processor.LBL_COL].nunique() < 2)
    print(groups)

    count = 0

    for uid, iid, history, candidate, click in tqdm(processor.generate(
            slicer=configuration.slicer,
            source=configuration.source
    )):
        # print(uid, iid, history, candidate, click)
        print(f'User: {uid}, Item: {iid}, History, Click: {click}')
        print(f'History:')
        for i, h in enumerate(history):
            print(f'\t{i:2d}: {h}')
        print(f'Candidate: {candidate}')

        count += 1
        if count > 10:
            break

    if processor.test_set_required:
        test_set_user = len(processor.test_set[processor.UID_COL].unique())
        pnt(f'Test set: {len(processor.test_set)} with {test_set_user} users')

    if processor.finetune_set_required:
        fine_tune_set_user = len(processor.finetune_set[processor.UID_COL].unique())
        pnt(f'Finetune set: {len(processor.finetune_set)} with {fine_tune_set_user} users')
