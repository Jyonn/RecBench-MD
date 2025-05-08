import os
import random
from typing import Type, cast

import pigmento
import torch
from pigmento import pnt
from tqdm import tqdm

from loader.class_hub import ClassHub
from loader.map import Map
from loader.seq_preparer import SeqPreparer
from sc_tuner import DiscreteCodeTuner
from seq_model.base_seqmodel import BaseSeqModel
from utils.code import get_code_indices
from utils.config_init import ConfigInit
from utils.function import load_seq_processor, seeding
from utils.seq_metrics import SeqMetricPool


class SeqTuner(DiscreteCodeTuner):
    PREPARER_CLASS = SeqPreparer

    num_codes: int
    caller: BaseSeqModel

    def load_processor(self, data):
        return load_seq_processor(data)

    def load_model(self):
        assert len(self.processors) == 1
        _, code_list, self.num_codes = get_code_indices(self.conf.code_path)

        models = ClassHub.seq_models()
        if self.model in models:
            model = models[self.model]  # type: Type[BaseSeqModel]
            assert issubclass(model, BaseSeqModel), f'{model} is not a subclass of BaseSeqModel'
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device(), num_codes=self.num_codes, code_list=code_list)
        raise ValueError(f'unknown model: {self.model}')

    def load_data(self):
        train_dfs, valid_dls = [], []

        assert self.train_processors[0].get_name() == self.valid_processors[0].get_name()

        processor = self.train_processors[0]
        preparer = self.PREPARER_CLASS(
            processor=processor,
            model=self.caller,
            conf=self.conf
        )
        if not preparer.has_generated:
            processor.load()
        train_dfs.append(preparer.load_or_generate(mode='train'))
        valid_dls.append(preparer.load_or_generate(mode='valid'))

        cast(BaseSeqModel, self.base_model).set_code_meta(preparer.code_tree, preparer.code_map)

        return train_dfs, valid_dls

    def _evaluate(self, dataloader, steps, metrics, step=1):
        # easy_decode = (is_test and self.conf.test_easy_decode) or (not is_test and self.conf.easy_decode)
        decode_mode = self.conf.decode_mode
        if decode_mode not in ['prod', 'easy', 'hard']:
            raise ValueError(f'unknown decode mode: {decode_mode}, should be one of prod, easy, hard')

        group_list, ranks_list = [], []
        item_index = 0
        for index, batch in tqdm(enumerate(dataloader), total=steps):
            if random.random() * step > 1:
                continue

            self.latency_timer.run('test')
            output = self.caller.decode(batch, width=self.conf.beam_width, decode_mode=decode_mode)
            self.latency_timer.run('test')
            if decode_mode == 'prod':
                rank = (cast(torch.Tensor, output) + 1).tolist()  # type: list
                batch_size = len(rank)
                for ib in range(batch_size):
                    local_rank = rank[ib]
                    group_list.extend([item_index] * len(local_rank))
                    ranks_list.extend(local_rank)
                    item_index += 1
            else:
                rank = output  # type: list
                batch_size = len(rank)
                groups = batch[Map.UID_COL].tolist()
                groups = groups[:batch_size]
                group_list.extend(groups)
                ranks_list.extend(rank)

        pool = SeqMetricPool.parse(metrics, num_items=self.num_codes, prod_mode=decode_mode == 'prod')
        return pool.calculate(ranks_list, group_list)

    def evaluate(self, valid_dls, epoch, metrics=None):
        total_valid_steps = self._get_steps(valid_dls)

        self.caller.model.eval()
        with torch.no_grad():
            pnt(f'(epoch {epoch}) validating dataset of {len(valid_dls)}: {self.valid_processors[0].get_name()}')
            results = self._evaluate(valid_dls[0], total_valid_steps[0], metrics=[self.conf.valid_metric], step=self.conf.valid_step)
            metric_name = list(results.keys())[0]
            metric_value = results[metric_name]
            pnt(f'(epoch {epoch}) validation on {self.valid_processors[0].get_name()} dataset with {metric_name}: {metric_value:.4f}')
        self.caller.model.train()

        action = self.monitor.push(metric_name, metric_value)
        if action is self.monitor.BEST:
            self.caller.save(os.path.join(self.log_dir, f'{self.sign}.pt'))
            pnt(f'saving best model to {self.log_dir}/{self.sign}.pt')
        return action

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

        cast(BaseSeqModel, self.base_model).set_code_meta(preparer.code_tree, preparer.code_map)

        total_valid_steps = self._get_steps([test_dl])

        self.caller.model.eval()
        with torch.no_grad():
            results = self._evaluate(test_dl, total_valid_steps[0], metrics=self.conf.metrics.split('+'))
            for metric, value in results.items():
                pnt(f'{metric}: {value:.4f}')

    def latency(self):
        self.latency_timer.activate()
        self.latency_timer.clear()

        self.load_data()
        processor = self.train_processors[0]
        preparer = self.PREPARER_CLASS(
            processor=processor,
            model=self.caller,
            conf=self.conf
        )
        if not preparer.has_generated:
            processor.load()
        test_dl = preparer.load_or_generate(mode='test')
        total_valid_steps = self._get_steps([test_dl])
        try:
            with torch.no_grad():
                self._evaluate(test_dl, total_valid_steps[0], metrics=self.conf.metrics.split('+'))
        except KeyboardInterrupt:
            pass

        st = self.latency_timer.status_dict['test']
        pnt(f'Total {st.count} steps, avg ms {st.avgms():.4f}')


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['model', 'data', 'code_path'],
        default_args=dict(
            mode='train',
            slicer=-20,
            gpu=None,
            valid_metric='NDCG@10',
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
            beam_width=20,
            prod_mode=0,
            seed=2024,
            decode_mode='prod',
            valid_step=1,
            align_step=1,
            metrics='+'.join(['NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20']),
        ),
        makedirs=[]
    ).parse()

    configuration.train = configuration.valid = configuration.data

    seeding(configuration.seed)

    tuner = SeqTuner(conf=configuration)
    tuner.run()
