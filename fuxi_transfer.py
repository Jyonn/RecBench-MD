import os

import polars as pl
import pandas as pd
import pigmento
from pigmento import pnt

from process.base_processor import BaseProcessor
from utils.config_init import ConfigInit
from utils.data import get_data_dir
from utils.function import load_sero_processor, load_processor


class Transfer:
    def __init__(self, conf):
        self.conf = conf

        data = self.conf.data.lower()
        data_dir = get_data_dir(data)
        load_processor_method = load_processor if not self.conf.sero else load_sero_processor
        self.processor = load_processor_method(data, data_dir=data_dir)  # type: BaseProcessor
        self.processor.load()

        self.valid_user_set = self.processor.load_valid_user_set(valid_ratio=self.conf.valid_ratio)
        self.export_dir = os.path.join('data', f'fuxi_{data}', f'{self.conf.valid_ratio}')
        os.makedirs(self.export_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.export_dir, 'train.csv')) and \
                os.path.exists(os.path.join(self.export_dir, 'valid.csv')) and \
                os.path.exists(os.path.join(self.export_dir, 'test.csv')):
            pnt('Data already exists, skipping...')
            exit()

    def load_datalist(self, source):
        datalist = []

        for uid, candidate, history, label in self.processor.generate(
            slicer=-100,
            source=source,
            id_only=True,
        ):
            datalist.append({
                'uid': uid,
                'iid': candidate,
                'history': '^'.join(history),
                'click': label,
            })
        return datalist

    def split_datalist(self, datalist):
        train_dl = []
        valid_dl = []
        for data in datalist:
            if data['uid'] in self.valid_user_set:
                valid_dl.append(data)
            else:
                train_dl.append(data)
        return train_dl, valid_dl

    def run(self):
        finetune_dl = self.load_datalist('finetune')
        train_dl, valid_dl = self.split_datalist(finetune_dl)
        test_dl = self.load_datalist('test')

        pnt(f'train: {len(train_dl)}, valid: {len(valid_dl)}, test: {len(test_dl)}')
        train_df = pd.DataFrame(train_dl)
        valid_df = pd.DataFrame(valid_dl)
        test_df = pd.DataFrame(test_dl)

        train_df.to_csv(os.path.join(self.export_dir, 'train.csv'), sep=',', index=False)
        valid_df.to_csv(os.path.join(self.export_dir, 'valid.csv'), sep=',', index=False)
        test_df.to_csv(os.path.join(self.export_dir, 'test.csv'), sep=',', index=False)


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data'],
        default_args=dict(
            valid_ratio=0.1,
            sero=False,
        ),
        makedirs=[]
    ).parse()

    transfer = Transfer(configuration)
    transfer.run()
