import os.path
from typing import Union, cast

import numpy as np
import pigmento
from pigmento import pnt

from loader.preparer import Preparer
from model.base_model import BaseModel
from model.bert_model import BertBaseModel, BertLargeModel
from model.lc_model import QWen2TH7BModel, GLM4TH9BModel, Mistral7BModel, Phi3TH7BModel
from model.llama_model import Llama1Model, Llama2Model
from model.opt_model import OPT1BModel, OPT350MModel
from model.p5_model import P5BeautyModel
from service.base_service import BaseService
from utils.config_init import ConfigInit
from utils.export import Exporter
from utils.function import load_processors
from utils.gpu import GPU
from utils.metrics import MetricPool
from utils.tqdm_printer import TqdmPrinter


class Tuner:
    def __init__(self, conf):
        self.models = [
            BertBaseModel, BertLargeModel,
            Llama1Model, Llama2Model,
            OPT1BModel, OPT350MModel,
            QWen2TH7BModel, GLM4TH9BModel, Mistral7BModel, Phi3TH7BModel,
            P5BeautyModel
        ]

        self.conf = conf
        self.model = conf.model.replace('.', '').lower()

        self.processors = [processor_class(cache=True) for processor_class in load_processors()]
        for processor in self.processors:
            processor.load()

        self.caller = self.load_model()  # type: Union[BaseService, BaseModel]
        self.use_service = isinstance(self.caller, BaseService)

        self.log_dir = os.path.join('tuning', self.model)

        os.makedirs(self.log_dir, exist_ok=True)
        pigmento.add_log_plugin(os.path.join(self.log_dir, f'{self.model}.log'))
        self.exporter = Exporter(os.path.join(self.log_dir, f'{self.model}.dat'))

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            pnt('Manually choosing CPU device')
            return 'cpu'
        pnt(f'Manually choosing {self.conf.gpu}-th GPU')
        return f'cuda:{self.conf.gpu}'

    def load_model(self):
        for model in self.models:
            if model.get_name() == self.model:
                pnt(f'loading {model.get_name()} model')
                return model(device=self.get_device())
        raise ValueError(f'Unknown model: {self.model}')

    @staticmethod
    def _get_steps(dataloaders: list):
        total_steps = []
        for dataloader in dataloaders:
            total_steps.append((len(dataloader.dataset) + dataloader.batch_size - 1) // dataloader.batch_size)
        return total_steps

    def finetune(self):
        self.caller = cast(BaseModel, self.caller)
        train_dls, valid_dls = [], []
        for processor in self.processors:
            train_dl, valid_dl = Preparer(processor, self.caller, self.conf).load_or_generate()
            train_dls.append(train_dl)
            valid_dls.append(valid_dl)

        total_train_steps, total_valid_steps = self._get_steps(train_dls), self._get_steps(valid_dls)

        for epoch in range(100):
            TqdmPrinter.activate()
            for i_dl, train_dl in enumerate(train_dls):
                for index, batch in enumerate(train_dl):
                    loss = self.caller.finetune(batch)
                    index += 1
                    pnt(f'(Epoch {epoch}) Training {i_dl}-th dataset {self.processors[i_dl].get_name()} of {len(train_dls)} with loss: {loss}', current=index, count=total_train_steps[i_dl])

            metric_name, metric_values = None, []
            for i_dl, valid_dl in enumerate(valid_dls):
                score_list, label_list, group_list = [], [], []
                for index, batch in enumerate(valid_dl):
                    scores = self.caller.evaluate(batch)
                    labels = batch['labels'].tolist()
                    groups = batch['uid'].tolist()

                    score_list.extend(scores)
                    label_list.extend(labels)
                    group_list.extend(groups)

                    pnt(f'(Epoch {epoch}) Validating {i_dl}-th dataset {self.processors[i_dl].get_name()} of {len(valid_dls)}', current=index + 1, count=total_valid_steps[i_dl])

                pool = MetricPool.parse([self.conf.valid_metric])
                results = pool.calculate(score_list, label_list, group_list)
                for k in results:
                    metric_name = k
                    metric_values.append(results[k])
                TqdmPrinter.deactivate()
                pnt(f'(Epoch {epoch}) End validation with {metric_name}: {np.mean(metric_values)}')

    def run(self):
        self.finetune()


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_basic_printer(TqdmPrinter())
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['model'],
        default_args=dict(
            slicer=-20,
            gpu=None,
            valid_metric='GAUC',
            valid_ratio=0.1,
            batch_size=64,
        ),
        makedirs=[]
    ).parse()

    tuner = Tuner(configuration)
    tuner.run()
