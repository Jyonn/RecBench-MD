import os.path
from typing import Union, Optional, List

import numpy as np
import pigmento
import torch
from pigmento import pnt

from model.base_model import BaseModel
from model.bert_model import BertBaseModel, BertLargeModel
from model.llama_model import Llama1Model, Llama2Model
from model.opt_model import OPT1BModel, OPT350MModel
from service.base_service import BaseService
from service.claude_service import Claude21Service, Claude3Service
from service.gpt_service import GPT4Service, GPT35Service
from utils.auth import GPT_KEY, CLAUDE_KEY
from utils.config_init import ConfigInit
from utils.export import Exporter
from utils.function import load_processor
from utils.gpu import GPU
from utils.metrics import MetricPool
from utils.tqdm_printer import TqdmPrinter

pigmento.add_time_prefix()
pnt.set_basic_printer(TqdmPrinter())


class Worker:
    def __init__(self, conf):
        self.services = [
            GPT35Service(auth=GPT_KEY), GPT4Service(auth=GPT_KEY),
            Claude21Service(auth=CLAUDE_KEY), Claude3Service(auth=CLAUDE_KEY)
        ]
        self.models = [BertBaseModel, BertLargeModel, Llama1Model, OPT1BModel, OPT350MModel, Llama2Model]

        self.conf = conf
        self.data = conf.data.lower()
        self.model = conf.model.replace('.', '').lower()

        self.type = conf.type
        assert self.type in ['prompt', 'embed'], f'Type {self.type} is not supported.'
        self.use_prompt = self.type == 'prompt'
        self.use_embed = self.type == 'embed'

        self.processor = load_processor(self.data)
        self.processor.load()
        self.caller = self.load_model_or_service()  # type: Union[BaseService, BaseModel]
        self.use_service = isinstance(self.caller, BaseService)

        self.log_dir = os.path.join('export', self.data)
        if self.use_embed:
            assert not self.use_service, 'Embedding is not supported for service.'
            self.log_dir = os.path.join('export', self.data + '_embed')

        os.makedirs(self.log_dir, exist_ok=True)
        pigmento.add_log_plugin(os.path.join(self.log_dir, f'{self.model}.log'))
        self.exporter = Exporter(os.path.join(self.log_dir, f'{self.model}.dat'))

    def load_model_or_service(self):
        for model in self.models:
            if model.get_name() == self.model:
                pnt(f'loading {model.get_name()} model')
                return model(device=GPU.auto_choose(torch_format=True))
        for service in self.services:
            if service.get_name() == self.model:
                pnt(f'loading {service.get_name()} service')
                return service
        raise ValueError(f'Unknown model/service: {self.model}')

    def test_prompt(self):
        input_template = """User behavior sequence: \n{0}\nCandidate item: {1}"""

        progress = self.exporter.load_progress()
        if progress > 0:
            pnt(f'directly start from {progress}')
        TqdmPrinter.activate()
        for index, data in enumerate(self.processor.generate(slicer=self.conf.slicer, source=self.conf.source)):
            if index < progress:
                continue

            uid, iid, history, candidate, click = data

            response: Optional[str, float] = None

            for _ in range(5):
                for i in range(len(history)):
                    _history = [f'({j + 1}) {history[i + j]}' for j in range(len(history) - i)]
                    input_sequence = input_template.format('\n'.join(_history), candidate)
                    response = self.caller(input_sequence)
                    if response is not None:
                        break
                if response is not None:
                    break
                candidate = candidate[:len(candidate) // 2]

            if response is None:
                pnt(f'failed to get response for {index} ({uid}, {iid})')
                self.exporter.save_progress(index)
                exit(0)

            pnt(f'Click: {click}, Response: {response}', current=index + 1, count=len(self.processor.test_set))
            self.exporter.write(response)
            self.exporter.save_progress(index + 1)
        TqdmPrinter.deactivate()

    def test_embed(self):
        history_template = """User behavior sequence: \n{0}"""
        candidate_template = """Candidate item: {0}"""

        progress = self.exporter.load_progress()
        if progress > 0:
            pnt(f'directly start from {progress}')
        item_dict = self.exporter.load_embed('item')
        user_dict = self.exporter.load_embed('user')

        TqdmPrinter.activate()
        for index, data in enumerate(self.processor.generate(slicer=self.conf.slicer, source=self.conf.source)):
            if index < progress:
                continue

            uid, iid, history, candidate, click = data

            user_embed: Optional[np.ndarray] = None
            if uid in user_dict:
                user_embed = user_dict[uid]
            else:
                for i in range(len(history)):
                    _history = [f'({j + 1}) {history[i + j]}' for j in range(len(history) - i)]
                    history_sequence = history_template.format('\n'.join(_history))
                    user_embed = self.caller.embed(history_sequence)
                    if user_embed is not None:
                        break
                if user_embed is None:
                    pnt(f'failed to get user embeds for {index} ({uid}, {iid})')
                    self.exporter.save_progress(index)
                    exit(0)
                user_dict[uid] = user_embed
                self.exporter.save_embed('user', user_dict)

            item_embed: Optional[np.ndarray] = None
            if iid in item_dict:
                item_embed = item_dict[iid]
            else:
                for _ in range(5):
                    candidate_sequence = candidate_template.format(candidate)
                    item_embed = self.caller.embed(candidate_sequence)
                    if item_embed is None:
                        candidate = candidate[:len(candidate) // 2]
                        continue
                if item_embed is None:
                    pnt(f'failed to get item embeds for {index} ({uid}, {iid})')
                    self.exporter.save_progress(index)
                    exit(0)
                item_dict[iid] = item_embed
                self.exporter.save_embed('item', item_dict)

            # score = torch.dot(item_embed, user_embed).item()
            # score = np.dot(item_embed, user_embed)
            # switch dot product to cosine similarity
            score = np.dot(item_embed, user_embed) / (np.linalg.norm(item_embed) * np.linalg.norm(user_embed))

            pnt(f'Click: {click}, Score: {score}', current=index + 1, count=len(self.processor.test_set))
            self.exporter.write(score)
            self.exporter.save_progress(index + 1)
        TqdmPrinter.deactivate()

    def evaluate(self):
        scores = self.exporter.read(from_convert=self.use_service)  # type: List[float]

        source_set = self.processor.get_source_set(self.conf.source)
        labels = source_set[self.processor.CLK_COL].values
        groups = source_set[self.processor.UID_COL].values

        pool = MetricPool.parse(self.conf.metrics.split('|'))
        results = pool.calculate(scores, labels, groups)
        for metric, value in results.items():
            pnt(f'{metric}: {value:.4f}')

        self.exporter.save_metrics(results)

    def auto_convert(self):
        progress = self.exporter.load_progress()
        source_set = self.processor.get_source_set(self.conf.source)
        assert progress == len(source_set), f'{self.conf.source} is not fully tested'

        labels = self.exporter.read(to_float=False)  # type: List[str]
        # use upper string
        labels = [label.upper().strip() for label in labels]
        scores = []

        rule_counts = [0, 0, 0]
        for label in labels:
            # Rule 1: fully YES/NO detect
            if label in ['YES', 'NO']:
                scores.append(1 if label == 'YES' else 0)
                rule_counts[0] += 1
                continue
            # Rule 2: partially YES/NO detect
            if ('YES' in label) ^ ('NO' in label):
                scores.append(1 if 'YES' in label else 0)
                rule_counts[1] += 1
                continue
            # Rule 3: manually detect
            pnt(f'Please manually convert: {label}')
            scores.append(int(input('Score (0/1): ')))
            rule_counts[2] += 1

        pnt(f'Rule 1: {rule_counts[0]}, Rule 2: {rule_counts[1]}, Rule 3: {rule_counts[2]}')
        self.exporter.save_convert(scores)

    def run(self):
        if self.use_prompt:
            self.test_prompt()
        else:
            self.test_embed()
        if self.use_service:
            self.auto_convert()
        self.evaluate()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            slicer=-20,
            source='test',
            metrics='|'.join(['GAUC', 'NDCG@1', 'NDCG@5', 'MRR', 'F1', 'Recall@1', 'Recall@5']),
            type='prompt',
        ),
        makedirs=[]
    ).parse()

    worker = Worker(configuration)
    worker.run()
