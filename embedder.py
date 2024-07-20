import os

import numpy as np
import pigmento
from pigmento import pnt
from tqdm import tqdm

from loader.class_hub import ClassHub
from process.base_processor import BaseProcessor
from utils.config_init import ConfigInit
from utils.function import load_processor
from utils.gpu import GPU


class Embedder:
    def __init__(self, conf):
        self.conf = conf

        self.data = conf.data.lower()
        self.model = conf.model.replace('.', '').lower()

        self.processor = load_processor(self.data)  # type: BaseProcessor
        self.processor.load()

        self.caller = self.load_model()

        self.log_dir = os.path.join('export', self.data)

        os.makedirs(self.log_dir, exist_ok=True)
        pigmento.add_log_plugin(os.path.join(self.log_dir, f'{self.model}-embedder.log'))

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            pnt('manually choosing CPU device')
            return 'cpu'
        pnt(f'manually choosing {self.conf.gpu}-th GPU')
        return f'cuda:{self.conf.gpu}'

    def load_model(self):
        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device())
        raise ValueError(f'unknown model: {self.model}')

    def embed(self):
        item_embeddings = []
        for item_id in tqdm(self.processor.item_vocab):
            item = self.processor.organize_item(item_id, item_attrs=self.processor.default_attrs)
            embedding = self.caller.embed(item).cpu().detach().numpy()
            item_embeddings.append(embedding)
        item_embeddings = np.array(item_embeddings)
        np.save(os.path.join(self.log_dir, f'{self.model}-embeds.npy'), item_embeddings)
        pnt(f'embeddings saved to {self.log_dir}/{self.model}-embeds.npy')

    def run(self):
        self.embed()


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            gpu=None,
            tuner=None,
            rerun=False,
        ),
        makedirs=[]
    ).parse()

    embedder = Embedder(configuration)
    embedder.run()
