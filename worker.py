from typing import Union

from model.base_model import BaseModel
from model.bert_model import BertBaseModel, BertLargeModel
from model.llama_model import Llama1Model
from model.opt_model import OPT1BModel, OPT350MModel
from process.mind_processor import MINDProcessor
from sdk.base_service import BaseService
from sdk.claude_service import Claude21Service, Claude3Service
from sdk.gpt_service import GPT4Service, GPT35Service
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
        self.device = GPU.auto_choose(torch_format=True)

        self.data = conf.data.lower()
        self.model = conf.model.lower()

        self.processor = self.load_processor()
        self.caller = self.load_model_or_service()  # type: Union[BaseService, BaseModel]
        self.use_service = isinstance(self.caller, BaseService)

    def load_processor(self):
        for dataset in self.datasets:
            if dataset.get_name() == self.data:
                return dataset()
        raise ValueError(f'Unknown dataset: {self.data}')

    def load_model_or_service(self):
        for model in self.models:
            if model.get_name() == self.model:
                return model(device=self.device)
        for service in self.services:
            if service.get_name() == self.model:
                return service
        raise ValueError(f'Unknown model/service: {self.model}')

    def run(self):
        for uid, iid, history, candidate, click in self.processor.iterate():
            pass


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(max_len=20),
        makedirs=[]
    ).parse()

    worker = Worker(configuration)
    worker.run()
