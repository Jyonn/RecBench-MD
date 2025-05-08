import os.path

import numpy as np
import pandas as pd
import pigmento
import torch
from pigmento import pnt
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.embed_preparer import EmbedPreparer
from loader.map import Map
from tuner import Tuner
from utils.config_init import ConfigInit
from utils.metrics import MetricPool


torch.autograd.set_detect_anomaly(True)


class EmbedTuner(Tuner):
    PREPARER_CLASS = EmbedPreparer

    @property
    def test_command(self):
        return f'python worker.py --model {self.model} --tuner {self.sign} --data <data_name> --type embed'

    def evaluate(self, valid_dls, epoch):
        total_valid_steps = self._get_steps(valid_dls)

        self.caller.model.eval()
        with torch.no_grad():
            metric_name, metric_values = None, []
            for i_dl, valid_dl in enumerate(valid_dls):
                pnt(f'(epoch {epoch}) validating {i_dl + 1}-th dataset of {len(valid_dls)}: {self.valid_processors[i_dl].get_name()}')
                score_list, label_list, group_list = [], [], []
                for index, batch in tqdm(enumerate(valid_dl), total=total_valid_steps[i_dl]):
                    scores = self.caller.evaluate_embed(batch)
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

        if epoch is None:
            return

        action = self.monitor.push(metric_name, metric_value)
        if action is self.monitor.BEST:
            self.caller.save(os.path.join(self.log_dir, f'{self.sign}.pt'))
            pnt(f'saving best model to {self.log_dir}/{self.sign}.pt')
        return action

    def test(self):
        test_dls = self.load_test_data()
        self.evaluate(test_dls, epoch=None)

    def dev(self):
        _, valid_dls = self.load_data()
        self.evaluate(valid_dls, epoch=None)

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
                loss = self.caller.finetune_embed(batch)
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

if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['model', 'train', 'valid'],
        default_args=dict(
            type='embed_tuner',
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
        ),
        makedirs=[]
    ).parse()

    tuner = EmbedTuner(configuration)
    tuner.run()
