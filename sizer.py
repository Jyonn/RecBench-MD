import pigmento
from pigmento import pnt
from torch import nn

from model.base_model import BaseModel
from model.bert_model import BertBaseModel, BertLargeModel
from model.lc_model import QWen2TH7BModel, GLM4TH9BModel, Mistral7BModel, Phi3TH7BModel
from model.llama_model import Llama1Model, Llama2Model
from model.opt_model import OPT1BModel, OPT350MModel
from model.p5_model import P5BeautyModel
from model.recformer_model import RecformerModel
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.tqdm_printer import TqdmPrinter


class Sizer:
    def __init__(self, conf):
        self.models = [
            BertBaseModel, BertLargeModel,
            Llama1Model, Llama2Model,
            OPT1BModel, OPT350MModel,
            QWen2TH7BModel, GLM4TH9BModel, Mistral7BModel, Phi3TH7BModel,
            P5BeautyModel, RecformerModel,
        ]

        self.conf = conf
        self.model = conf.model.replace('.', '').lower()

        self.caller = self.load_model()  # type: BaseModel

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
        raise ValueError(f'Unknown model: {self.model}')

    def run(self):
        model = self.caller.model  # type: nn.Module
        num_parameters = 0
        for name, param in model.named_parameters():
            num_parameters += param.numel()
        pnt(f'Total number of parameters: {num_parameters/1e6:.0f}M')


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
            gpu=None,
        ),
        makedirs=[]
    ).parse()

    sizer = Sizer(configuration)
    sizer.run()
