import sys
from typing import Type, cast

import pigmento
import torch
from pigmento import pnt
from tqdm import tqdm

from loader.class_hub import ClassHub
from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.map import Map
from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.base_model import BaseModel
from process.base_processor import BaseProcessor
from tuner import Tuner
from utils.code import get_code_indices
from utils.config_init import ConfigInit
from utils.function import seeding, load_processor, load_sero_processor
from utils.metrics import MetricPool


class DiscreteCodeTuner(Tuner):
    PREPARER_CLASS = DiscreteCodePreparer

    num_codes: int

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #
    #     if not self.conf.gist:
    #         self.caller = cast(BaseDiscreteCodeModel, self.caller)
    #         self.caller.load_from_gist(self.conf.gist)

    def load_processor(self, data):
        return load_sero_processor(data) if self.conf.sero else load_processor(data)  # type: BaseProcessor

    def load_model(self):
        assert len(self.processors) == 1
        _, _, self.num_codes = get_code_indices(self.conf.code_path)

        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]  # type: Type[BaseModel]
            assert issubclass(model, BaseDiscreteCodeModel), f'{model} is not a subclass of BaseDiscreteCodeModel'
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device(), num_codes=self.num_codes)
        raise ValueError(f'unknown model: {self.model}')

    def test(self):
        assert self.train_processors[0].get_name() == self.valid_processors[0].get_name()

        processor = self.train_processors[0]
        preparer = self.PREPARER_CLASS(
            processor=processor,
            model=self.caller,
            conf=self.conf
        )
        if not preparer.has_generated:
            processor.load()
        preparer.load_or_generate(mode='train')
        test_dl = preparer.load_or_generate(mode='test')

        total_valid_steps = self._get_steps([test_dl])

        self.caller.model.eval()
        with torch.no_grad():
            score_list, label_list, group_list = [], [], []
            for index, batch in tqdm(enumerate(test_dl), total=total_valid_steps[0]):
                scores = self.caller.evaluate(batch)
                labels = batch[Map.LBL_COL].tolist()
                groups = batch[Map.UID_COL].tolist()

                score_list.extend(scores)
                label_list.extend(labels)
                group_list.extend(groups)

            pool = MetricPool.parse(self.conf.metrics.split('+'))
            results = pool.calculate(score_list, label_list, group_list)
            for metric, value in results.items():
                pnt(f'{metric}: {value:.4f}')

    def run(self):
        if self.conf.mode == 'train':
            self.finetune()
        elif self.conf.mode == 'test':
            self.test()

    @property
    def test_command(self):
        arg = ' '.join(sys.argv).replace('--mode train', '')
        return f'python {arg} --tuner {self.sign} --mode test'


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    seeding(2024)

    configuration = ConfigInit(
        required_args=['model', 'data', 'code_path'],
        default_args=dict(
            gist=None,
            sero=False,
            mode='train',
            slicer=-20,
            gpu=None,
            valid_metric='GAUC',
            valid_ratio=0.1,
            batch_size=32,
            use_lora=True,
            lora_r=32,
            lora_alpha=128,
            lora_dropout=0.1,
            lr=0.00001,
            acc_batch=1,
            eval_interval=0,
            patience=2,
            tuner=None,
            init_eval=True,
            metrics='+'.join(['GAUC', 'NDCG@1', 'NDCG@5', 'MRR', 'F1', 'Recall@1', 'Recall@5']),
            alignment=True,
        ),
        makedirs=[]
    ).parse()

    configuration.train = configuration.valid = configuration.data

    tuner = DiscreteCodeTuner(conf=configuration)
    tuner.run()
