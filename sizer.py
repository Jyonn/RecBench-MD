import pigmento
from pigmento import pnt
from torch import nn

from loader.class_hub import ClassHub
from model.base_model import BaseModel
from utils.config_init import ConfigInit
from utils.tqdm_printer import TqdmPrinter


class Sizer:
    def __init__(self, conf):
        self.models = ClassHub.models()

        self.conf = conf
        self.model = conf.model.replace('.', '').lower()

        self.caller = self.load_model()  # type: BaseModel

    def load_model(self):
        if self.model in self.models:
            model = self.models[self.model]
            pnt(f'loading {model.get_name()} model')
            return model(device='cpu')
        raise ValueError(f'unknown model: {self.model}')

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
        default_args=dict(),
        makedirs=[]
    ).parse()

    sizer = Sizer(configuration)
    sizer.run()
