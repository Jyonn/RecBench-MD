import os.path
from typing import Union, Optional

import pigmento
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

        self.processor = load_processor(self.data)
        self.processor.load()
        self.caller = self.load_model_or_service()  # type: Union[BaseService, BaseModel]
        self.use_service = isinstance(self.caller, BaseService)

        self.log_dir = os.path.join('export', self.data)
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

    def test(self):
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

            pnt(f'({index + 1}/{len(self.processor.test_set)}) Click: {click}, Response: {response}')
            self.exporter.write(response)
            self.exporter.save_progress(index + 1)
        TqdmPrinter.deactivate()

    def evaluate(self):
        scores = self.exporter.read()

        source_set = self.processor.get_source_set(self.conf.source)
        labels = source_set[self.processor.CLK_COL].values
        groups = source_set[self.processor.UID_COL].values

        pool = MetricPool.parse(self.conf.metrics.split('|'))
        results = pool.calculate(scores, labels, groups)
        for metric, value in results.items():
            pnt(f'{metric}: {value:.4f}')

        self.exporter.save_metrics(results)

    def run(self):
        self.test()
        self.evaluate()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            slicer=-20,
            source='test',
            metrics='|'.join(['GAUC', 'NDCG@1', 'NDCG@5', 'MRR', 'F1', 'Recall@1', 'Recall@5']),
        ),
        makedirs=[]
    ).parse()

    worker = Worker(configuration)
    worker.run()
