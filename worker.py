from typing import Union, Optional

from pigmento import pnt

from model.base_model import BaseModel
from model.bert_model import BertBaseModel, BertLargeModel
from model.llama_model import Llama1Model
from model.opt_model import OPT1BModel, OPT350MModel
from process.mind_processor import MINDProcessor
from service.base_service import BaseService
from service.claude_service import Claude21Service, Claude3Service
from service.gpt_service import GPT4Service, GPT35Service
from utils.auth import GPT_KEY, CLAUDE_KEY
from utils.config_init import ConfigInit
from utils.gpu import GPU


class Worker:
    def __init__(self, conf):
        self.datasets = [MINDProcessor]
        self.services = [
            GPT35Service(auth=GPT_KEY), GPT4Service(auth=GPT_KEY),
            Claude21Service(auth=CLAUDE_KEY), Claude3Service(auth=CLAUDE_KEY)
        ]
        self.models = [BertBaseModel, BertLargeModel, Llama1Model, OPT1BModel, OPT350MModel]

        self.conf = conf
        self.data = conf.data.lower()
        self.model = conf.model.replace('.', '').lower()

        self.processor = self.load_processor()
        self.processor.load()
        self.caller = self.load_model_or_service()  # type: Union[BaseService, BaseModel]
        self.use_service = isinstance(self.caller, BaseService)

        self.device = None if self.use_service else GPU.auto_choose(torch_format=True)

    def load_processor(self):
        for dataset in self.datasets:
            if dataset.get_name() == self.data:
                pnt(f'loading {dataset.get_name()} processor')
                return dataset(data_dir=self.conf.data_dir)
        raise ValueError(f'Unknown dataset: {self.data}')

    def load_model_or_service(self):
        for model in self.models:
            if model.get_name() == self.model:
                pnt(f'loading {model.get_name()} model')
                return model(device=self.device)
        for service in self.services:
            if service.get_name() == self.model:
                pnt(f'loading {service.get_name()} service')
                return service
        raise ValueError(f'Unknown model/service: {self.model}')

    def run(self):
        count = 0
        input_template = """User behavior sequence: {0}\n Candidate item: {1}"""
        for uid, iid, history, candidate, click in self.processor.generate(slicer=self.conf.slicer):
            # print(f'User: {uid}, Item: {iid}, History, Click: {click}')
            # print(f'History:')
            # for i, h in enumerate(history):
            #     print(f'\t{i:2d}: {h}')
            # print(f'Candidate: {candidate}')
            # (1) The Difference Between Green and Orange Antifreeze
            # (2) Road built by biblical villain uncovered in Jerusalem
            # (3) Boat inches closer to Niagara Falls edge after being grounded for century
            # (4) 24 Ways to Shrink Your Belly in 24 Hours
            # Candidate item: UFC Tampa results: Jedrzejczyk dominates Waterson

            for i in range(len(history)):
                history[i] = f'({i + 1}) {history[i]}'

            response: Optional[str, float] = None

            for i in range(len(history), 1, -1):
                input_sequence = input_template.format('\n'.join(history[:i]), candidate)
                response = self.caller(input_sequence)
                if response is not None:
                    break
            if response is None:
                pnt(uid, iid)
            pnt(f'Click: {click}, Response: {response}')

            count += 1
            if count > 10:
                break


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(slicer=-10, data_dir=None),
        makedirs=[]
    ).parse()

    worker = Worker(configuration)
    worker.run()
