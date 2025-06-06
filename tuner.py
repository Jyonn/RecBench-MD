import copy
import hashlib
import json
import os.path
import random
import sys

import numpy as np
import pandas as pd
import pigmento
import torch
from oba import Obj
from pigmento import pnt
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.class_hub import ClassHub
from loader.discrete_code_preparer import DiscreteCodePreparer
from loader.map import Map
from loader.preparer import Preparer
from model.base_model import BaseModel
from process.base_processor import BaseProcessor
from utils.config_init import ConfigInit
from utils.function import load_processor
from utils.gpu import GPU
from utils.metrics import MetricPool
from utils.monitor import Monitor
from utils.timer import Timer


class Tuner:
    PREPARER_CLASS = Preparer

    def __init__(self, conf):
        self.conf = conf
        self.model = conf.model.replace('.', '').lower()
        self.train_data = conf.train.split('+')
        self.valid_data = conf.valid.split('+')

        self.processors = dict()
        for data in set(self.train_data + self.valid_data):
            self.processors[data] = self.load_processor(data)  # type: BaseProcessor
        self.train_processors = [self.processors[data] for data in self.train_data]  # type: list[BaseProcessor]
        self.valid_processors = [self.processors[data] for data in self.valid_data]  # type: list[BaseProcessor]

        if self.conf.tuner is not None:
            self.conf.tuner = str(self.conf.tuner).replace('@', '')
            self.conf.tuner = os.path.join('tuning', self.model, self.conf.tuner + '.json')
            self.tuner_meta = Obj(json.load(open(self.conf.tuner)))
            required_args = ['use_lora', 'lora_r', 'lora_alpha', 'lora_dropout']
            for arg in required_args:
                assert arg in self.tuner_meta, f'{arg} is required in tuner configuration'
                assert self.tuner_meta[arg] == self.conf[arg], f'{arg} should be consistent with previous tuner'

        self.log_dir = os.path.join('tuning', self.model)
        os.makedirs(self.log_dir, exist_ok=True)

        pnt(f'python {" ".join(sys.argv)}')

        self.meta = self.get_meta()
        self.sign = self.get_signature()

        self.model_path = os.path.join(self.log_dir, f'{self.sign}.pt')
        self.meta_path = os.path.join(self.log_dir, f'{self.sign}.json')
        self.log_path = os.path.join(self.log_dir, f'{self.sign}.log')
        pigmento.add_log_plugin(self.log_path)

        self.base_model = self.caller = self.load_model()  # type: BaseModel
        self.caller.prepare_model_finetuning(self.conf, inference_mode=False, tune_from=self.conf.tune_from)
        if self.conf.tuner:
            self.caller.load(self.conf.tuner.replace('.json', '.pt'))
        self.caller.post_init()

        json.dump(self.meta, open(self.meta_path, 'w'), indent=2)

        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.caller.model.parameters()),
            lr=self.conf.lr
        )

        self.monitor = Monitor(patience=self.conf.patience)
        self.latency_timer = Timer(activate=False)

    def load_processor(self, data):
        return load_processor(data)

    @property
    def test_command(self):
        return f'python worker.py --model {self.model} --tuner {self.sign} --data <data_name>'

    def get_meta(self):
        conf = copy.deepcopy(Obj.raw(self.conf))
        conf['train'] = '+'.join(sorted(self.train_data))
        conf['valid'] = '+'.join(sorted(self.valid_data))
        conf['model'] = self.model
        del conf['gpu']
        del conf['init_eval']
        conf['use_lora'] = int(conf['use_lora'])
        return conf

    def get_signature(self):
        keys = sorted(self.meta.keys())
        key = '-'.join([f'{k}={self.meta[k]}' for k in keys])
        md5 = hashlib.md5(key.encode()).hexdigest()
        return md5[:6]

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            pnt('manually choosing CPU device')
            return 'cpu'

        pnt(f'manually choosing {self.conf.gpu}-th GPU')
        if isinstance(self.conf.gpu, int):
            return f'cuda:{self.conf.gpu}'
        gpus = list(map(int, self.conf.gpu.split('+')))
        return f'cuda:{gpus[0]}', gpus

    def load_model(self):
        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]
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
                pnt(f'(epoch {epoch}) validating {i_dl + 1}-th dataset of {len(valid_dls)}: {self.valid_processors[i_dl].get_name()}')
                score_list, label_list, group_list = [], [], []
                for index, batch in tqdm(enumerate(valid_dl), total=total_valid_steps[i_dl]):
                    self.latency_timer.run('test')
                    scores = self.caller.evaluate(batch)
                    self.latency_timer.run('test')
                    labels = batch[Map.LBL_COL].tolist()
                    groups = batch[Map.UID_COL].tolist()

                    score_list.extend(scores)
                    label_list.extend(labels)
                    group_list.extend(groups)

                pool = MetricPool.parse([self.conf.valid_metric])
                results = pool.calculate(score_list, label_list, group_list)
                for k in results:
                    metric_name = k
                    metric_values.append(results[k])
                pnt(f'(epoch {epoch}) validation on {self.valid_processors[i_dl].get_name()} dataset with {metric_name}: {metric_values[-1]:.4f}')
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

    def get_eval_interval(self, total_train_steps):
        if self.conf.eval_interval == 0:
            self.conf.eval_interval = -1

        if self.conf.eval_interval < 0:
            return total_train_steps // -self.conf.eval_interval

        return self.conf.eval_interval

    def load_data(self):
        train_dfs, valid_dls = [], []

        for processor in self.train_processors:
            preparer = self.PREPARER_CLASS(
                processor=processor,
                model=self.caller,
                conf=self.conf
            )
            if not preparer.has_generated:
                processor.load()
            train_dfs.append(preparer.load_or_generate(mode='train'))

        for processor in self.valid_processors:
            preparer = self.PREPARER_CLASS(
                processor=processor,
                model=self.caller,
                conf=self.conf
            )
            if not preparer.has_generated:
                processor.load()
            valid_dls.append(preparer.load_or_generate(mode='valid'))

        return train_dfs, valid_dls

    def load_test_data(self):
        test_dls = []
        for processor in self.valid_processors:
            preparer = self.PREPARER_CLASS(
                processor=processor,
                model=self.caller,
                conf=self.conf
            )
            if not preparer.has_generated:
                processor.load()
            test_dls.append(preparer.load_or_generate(mode='test'))
        return test_dls

    def alignment(self):
        if not issubclass(self.PREPARER_CLASS, DiscreteCodePreparer):
            return

        if not self.conf.alignment:
            return

        if hasattr(self, 'alignment_train_dl') and hasattr(self, 'alignment_total_train_steps'):
            train_dl = self.alignment_train_dl
            total_train_steps = self.alignment_total_train_steps
        else:
            processor = self.train_processors[0]
            preparer = self.PREPARER_CLASS(
                processor=processor,
                model=self.caller,
                conf=self.conf
            )
            train_df = preparer.generate_item_alignment_data()
            train_ds = self.PREPARER_CLASS.DATASET_CLASS(train_df)
            train_dl = DataLoader(train_ds, batch_size=self.conf.batch_size, shuffle=True)
            self.__setattr__('alignment_train_dl', train_dl)
            total_train_steps = (len(train_ds) + self.conf.batch_size - 1) // self.conf.batch_size
            self.__setattr__('alignment_total_train_steps', total_train_steps)

        self.caller.model.train()
        accumulate_step = 0
        self.optimizer.zero_grad()
        for index, batch in tqdm(enumerate(train_dl), total=total_train_steps):
            if random.random() * self.conf.align_step >= 1:
                continue

            loss = self.caller.finetune(batch, alignment=True)
            loss.backward()

            accumulate_step += 1
            if accumulate_step == self.conf.acc_batch:
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulate_step = 0

    def finetune(self):
        train_dfs, valid_dls = self.load_data()

        train_dfs = pd.concat(train_dfs)
        train_ds = self.PREPARER_CLASS.DATASET_CLASS(train_dfs)
        train_ds.align(batch_size=self.conf.batch_size, ascending=False)
        train_dl = DataLoader(train_ds, batch_size=self.conf.batch_size, shuffle=False)

        total_train_steps = (len(train_ds) + self.conf.batch_size - 1) // self.conf.batch_size

        self.list_tunable_parameters()

        eval_interval = self.get_eval_interval(total_train_steps)

        if self.conf.init_eval:
            self.evaluate(valid_dls, -1)

        epoch = 0
        while True:
            self.caller.model.train()
            self.alignment()

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

                if (index + 1) % eval_interval == 0:
                    action = self.evaluate(valid_dls, epoch)
                    if action is self.monitor.STOP:
                        pnt('early stopping')
                        pnt(f'please evaluate the model by: {self.test_command}')
                        return

            epoch += 1

    def latency(self):
        self.latency_timer.activate()
        self.latency_timer.clear()
        train_dfs, valid_dls = self.load_data()

        try:
            self.evaluate(valid_dls, 0)
        except KeyboardInterrupt:
            pass
        st = self.latency_timer.status_dict['test']
        pnt(f'Total {st.count} steps, avg ms {st.avgms():.4f}')

    def test(self):
        raise NotImplemented

    def dev(self):
        raise NotImplemented

    def run(self):
        if self.conf.latency:
            self.latency()
            return
        if self.conf.mode == 'finetune':
            self.finetune()
        if self.conf.mode == 'dev':
            self.dev()
        if self.conf.mode == 'test':
            self.test()


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['model', 'train', 'valid'],
        default_args=dict(
            type='tuner',
            mode='finetune',
            slicer=-20,
            gpu=None,
            valid_metric='GAUC',
            valid_ratio=0.1,
            batch_size=32,
            use_lora=True,
            lora_r=32,
            lora_alpha=128,
            lora_dropout=0.1,
            lr=0.0001,
            acc_batch=1,
            eval_interval=0,
            patience=2,
            tuner=None,
            init_eval=False,
            align_step=1.0,
            tune_from=0,
        ),
        makedirs=[]
    ).parse()

    tuner = Tuner(configuration)
    tuner.run()
