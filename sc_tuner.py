from typing import Type

import pigmento
from pigmento import pnt

from loader.class_hub import ClassHub
from loader.discrete_code_preparer import DiscreteCodePreparer
from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.base_model import BaseModel
from tuner import Tuner
from utils.code import get_code_indices
from utils.config_init import ConfigInit


class DiscreteCodeTuner(Tuner):
    PREPARER_CLASS = DiscreteCodePreparer

    num_codes: int

    def load_model(self):
        assert len(self.processors) == 1
        _, self.num_codes = get_code_indices(self.conf.code_path)

        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]  # type: Type[BaseModel]
            assert issubclass(model, BaseDiscreteCodeModel), f'{model} is not a subclass of BaseDiscreteCodeModel'
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device(), num_codes=self.num_codes)
        raise ValueError(f'unknown model: {self.model}')


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['model', 'train', 'valid', 'code_path'],
        default_args=dict(
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
        ),
        makedirs=[]
    ).parse()

    tuner = DiscreteCodeTuner(conf=configuration)
    tuner.run()
