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


pigmento.add_time_prefix()


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

    def run(self):
        input_template = """User behavior sequence: \n{0}\nCandidate item: {1}"""

        progress = self.exporter.load_progress()
        pnt(f'directly start from {progress}')
        for index, data in enumerate(self.processor.generate(slicer=self.conf.slicer)):
            if index < progress:
                continue

            uid, iid, history, candidate, click = data

            for i in range(len(history)):
                history[i] = f'({i + 1}) {history[i]}'

            response: Optional[str, float] = None

            for i in range(len(history)):
                input_sequence = input_template.format('\n'.join(history[i:]), candidate)
                response = self.caller(input_sequence)
                if response is not None:
                    break
            if response is None:
                pnt(f'failed to get response for {index} ({uid}, {iid})')
                self.exporter.save_progress(index)
                exit(0)

            pnt(f'({index}/{len(self.processor.test_set)}) Click: {click}, Response: {response}')
            self.exporter.write(response)
            self.exporter.save_progress(index + 1)


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(slicer=-20),
        makedirs=[]
    ).parse()

    worker = Worker(configuration)
    worker.run()
