import copy
import hashlib
import json
import os.path
from typing import Type

import numpy as np
import pandas as pd
import pigmento
import torch
from oba import Obj
from pigmento import pnt
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.class_hub import ClassHub
from loader.dataset import Dataset
from loader.map import Map
from loader.preparer import Preparer
from model.base_dense_code_model import BaseDenseCodeModel
from model.base_model import BaseModel
from process.base_processor import BaseProcessor
from tuner import Tuner
from utils.config_init import ConfigInit
from utils.function import load_processor
from utils.gpu import GPU
from utils.metrics import MetricPool
from utils.monitor import Monitor
from utils.tqdm_printer import TqdmPrinter


class DenseCodeTuner(Tuner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_model(self):
        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]  # type: Type[BaseModel]
            assert issubclass(model, BaseDenseCodeModel), f'{model} is not a subclass of BaseDenseCodeModel'
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device())
        raise ValueError(f'unknown model: {self.model}')

    @staticmethod
    def _get_steps(dataloaders: list):
        total_steps = []
        for dataloader in dataloaders:
            total_steps.append((len(dataloader.dataset) + dataloader.batch_size - 1) // dataloader.batch_size)
        return total_steps

    def list_tunable_parameters(self):
        pnt('tunable parameters:')
        for name, param in self.caller.model.named_parameters():
            if param.requires_grad:
                pnt(f'{name}: {param.size()}')

    def evaluate(self, valid_dls, epoch):
        total_valid_steps = self._get_steps(valid_dls)

        self.caller.model.eval()
        with torch.no_grad():
            metric_name, metric_values = None, []
            for i_dl, valid_dl in enumerate(valid_dls):
                TqdmPrinter.activate()
                score_list, label_list, group_list = [], [], []
                for index, batch in enumerate(valid_dl):
                    scores = self.caller.evaluate(batch)
                    labels = batch[Map.LBL_COL].tolist()
                    groups = batch[Map.UID_COL].tolist()

                    score_list.extend(scores)
                    label_list.extend(labels)
                    group_list.extend(groups)

                    pnt(f'(epoch {epoch}) validating {i_dl + 1}-th dataset of {len(valid_dls)}: {self.valid_processors[i_dl].get_name()}',
                        current=index + 1, count=total_valid_steps[i_dl])

                pool = MetricPool.parse([self.conf.valid_metric])
                results = pool.calculate(score_list, label_list, group_list)
                for k in results:
                    metric_name = k
                    metric_values.append(results[k])
                pnt(f'(epoch {epoch}) validation on {self.valid_processors[i_dl].get_name()} dataset with {metric_name}: {metric_values[-1]:.4f}',
                    current=i_dl + 1, count=len(valid_dls))
                TqdmPrinter.deactivate()
        self.caller.model.train()

        metric_value = np.mean(metric_values).item()
        pnt(f'(epoch {epoch}) validation on all datasets with {metric_name}: {metric_value:.4f}')

        action = self.monitor.push(metric_name, metric_value)
        if action is self.monitor.BEST:
            self.caller.save(os.path.join(self.log_dir, f'{self.sign}.pt'))
            pnt(f'saving best model to {self.log_dir}/{self.sign}.pt')
        # elif action is self.monitor.STOP:
        #     pnt('early stopping')
        #     break
        return action

    def finetune(self):
        train_dfs, valid_dls = [], []

        for processor in self.train_processors:
            preparer = Preparer(processor, self.caller, self.conf)
            if not preparer.has_generated:
                processor.load()
            train_dfs.append(preparer.load_or_generate(mode='train'))

        for processor in self.valid_processors:
            preparer = Preparer(processor, self.caller, self.conf)
            if not preparer.has_generated:
                processor.load()
            valid_dls.append(preparer.load_or_generate(mode='valid'))

        train_dfs = pd.concat(train_dfs)
        train_ds = Dataset(train_dfs)
        train_ds.align(batch_size=self.conf.batch_size, ascending=False)
        train_dl = DataLoader(train_ds, batch_size=self.conf.batch_size, shuffle=False)

        total_train_steps = (len(train_ds) + self.conf.batch_size - 1) // self.conf.batch_size

        self.list_tunable_parameters()

        if self.conf.eval_interval < 0:
            self.conf.eval_interval = total_train_steps // -self.conf.eval_interval

        if self.conf.init_eval:
            self.evaluate(valid_dls, -1)

        for epoch in range(100):
            self.caller.model.train()
            accumulate_step = 0
            self.optimizer.zero_grad()
            for index, batch in tqdm(enumerate(train_dl), total=total_train_steps):
                loss = self.caller.finetune(batch)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == self.conf.acc_batch:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0

                if (index + 1) % self.conf.eval_interval == 0:
                    action = self.evaluate(valid_dls, epoch)
                    if action is self.monitor.STOP:
                        pnt('early stopping')
                        pnt(f'please evaluate the model by: python worker.py --model {self.model} --tuner {self.sign} --data <data_name>')
                        return
                # pnt(f'(epoch {epoch}), current loss: {loss.item():.4f}', current=index, count=total_train_steps)

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
        required_args=['model', 'train', 'valid'],
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
            eval_interval=1,
            patience=2,
            tuner=None,
            init_eval=True,
        ),
        makedirs=[]
    ).parse()

    tuner = DenseCodeTuner(configuration)
    tuner.run()
