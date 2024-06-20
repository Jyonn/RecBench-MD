import os.path

import numpy as np
import pandas as pd
import pigmento
import torch
from pigmento import pnt
from torch.utils.data import DataLoader

from loader.dataset import Dataset
from loader.map import Map
from loader.preparer import Preparer
from model.base_model import BaseModel
from model.bert_model import BertBaseModel, BertLargeModel
from model.lc_model import QWen2TH7BModel, GLM4TH9BModel, Mistral7BModel, Phi3TH7BModel
from model.llama_model import Llama1Model, Llama2Model
from model.opt_model import OPT1BModel, OPT350MModel
from model.p5_model import P5BeautyModel
from utils.config_init import ConfigInit
from utils.export import Exporter
from utils.function import load_processor
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
        self.data = conf.data.split('|')

        self.processors = []
        for data in self.data:
            self.processors.append(load_processor(data).load())

        self.caller = self.load_model()  # type: BaseModel
        self.caller.prepare_model_finetuning(self.conf)

        self.log_dir = os.path.join('tuning', self.model)

        os.makedirs(self.log_dir, exist_ok=True)
        pigmento.add_log_plugin(os.path.join(self.log_dir, f'{self.model}.log'))
        self.exporter = Exporter(os.path.join(self.log_dir, f'{self.model}.dat'))

        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.caller.model.parameters()),
            lr=self.conf.lr
        )

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            pnt('manually choosing CPU device')
            return 'cpu'
        pnt(f'manually choosing {self.conf.gpu}-th GPU')
        return f'cuda:{self.conf.gpu}'

    def load_model(self):
        for model in self.models:
            if model.get_name() == self.model:
                pnt(f'loading {model.get_name()} model')
                return model(device=self.get_device())
        raise ValueError(f'unknown model: {self.model}')

    @staticmethod
    def _get_steps(dataloaders: list):
        total_steps = []
        for dataloader in dataloaders:
            total_steps.append((len(dataloader.dataset) + dataloader.batch_size - 1) // dataloader.batch_size)
        return total_steps

    def finetune(self):
        train_dfs, valid_dls = [], []
        for processor in self.processors:
            train_df, valid_dl = Preparer(processor, self.caller, self.conf).load_or_generate()
            train_dfs.append(train_df)
            valid_dls.append(valid_dl)
        # concat all the dataframes
        train_dfs = pd.concat(train_dfs)
        train_ds = Dataset(train_dfs)
        train_dl = DataLoader(train_ds, batch_size=self.conf.batch_size, shuffle=True)

        total_train_steps = (len(train_ds) + self.conf.batch_size - 1) // self.conf.batch_size
        total_valid_steps = self._get_steps(valid_dls)

        for epoch in range(100):
            TqdmPrinter.activate()
            for index, batch in enumerate(train_dl):
                self.optimizer.zero_grad()
                loss = self.caller.finetune(batch)
                loss.backward()
                self.optimizer.step()

                index += 1
                pnt(f'(Epoch {epoch}), current loss: {loss.item():.4f}', current=index, count=total_train_steps)
            TqdmPrinter.deactivate()

            # self.caller.save('test.pth')

            metric_name, metric_values = None, []
            for i_dl, valid_dl in enumerate(valid_dls):
                TqdmPrinter.activate()
                score_list, label_list, group_list = [], [], []
                for index, batch in enumerate(valid_dl):
                    scores = self.caller.evaluate(batch)
                    labels = batch[Map.LBL_COl].tolist()
                    groups = batch[Map.UID_COL].tolist()

                    score_list.extend(scores)
                    label_list.extend(labels)
                    group_list.extend(groups)

                    pnt(f'(Epoch {epoch}) Validating {i_dl}-th dataset {self.processors[i_dl].get_name()} of {len(valid_dls)}', current=index + 1, count=total_valid_steps[i_dl])

                pool = MetricPool.parse([self.conf.valid_metric])
                results = pool.calculate(score_list, label_list, group_list)
                for k in results:
                    metric_name = k
                    metric_values.append(results[k])
                pnt(f'(Epoch {epoch}) End validation with {metric_name}: {np.mean(metric_values):.4f}', current=i_dl + 1, count=len(valid_dls))
                TqdmPrinter.deactivate()

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
            batch_size=32,
            data='mind|microlens|movielens',
            use_lora=True,
            lora_r=32,
            lora_alpha=128,
            lora_dropout=0.1,
            lr=0.0001,
        ),
        makedirs=[]
    ).parse()

    tuner = Tuner(configuration)
    tuner.run()
