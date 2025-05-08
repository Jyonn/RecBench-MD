import os

import numpy as np
import pigmento
from pigmento import pnt
from tqdm import tqdm

from loader.class_hub import ClassHub
from model.base_model import BaseModel
from process.base_processor import BaseProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor
from utils.config_init import ConfigInit
from utils.function import load_processor, load_seq_processor
from utils.gpu import GPU


class Embedder:
    def __init__(self, conf):
        self.conf = conf

        self.data = conf.data.lower()
        self.model = conf.model.replace('.', '').lower()

        if self.conf.attrs is None:
            self.attrs = None
        else:
            self.attrs = self.conf.attrs.split('+')

        if self.conf.seq:
            self.processor = load_seq_processor(self.data)  # type: BaseSeqProcessor
        else:
            self.processor = load_processor(self.data)  # type: BaseProcessor
        self.type = '.seq' if self.conf.seq else ''
        self.processor.load()

        self.caller = self.load_model()

        self.log_dir = os.path.join('export', self.data)

        os.makedirs(self.log_dir, exist_ok=True)
        pigmento.add_log_plugin(os.path.join(self.log_dir, f'{self.model}-embedder.log'))

        self.embedding_path = os.path.join(self.log_dir, f'{self.model}-embeds{self.type}.npy')

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            pnt('manually choosing CPU device')
            return 'cpu'
        pnt(f'manually choosing {self.conf.gpu}-th GPU')
        return f'cuda:{self.conf.gpu}'

    def load_model(self) -> BaseModel:
        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device()).post_init()
        raise ValueError(f'unknown model: {self.model}')

    def embed(self):
        item_embeddings = []
        for item_id in tqdm(self.processor.item_vocab):
            item = self.processor.organize_item(item_id, item_attrs=self.attrs or self.processor.default_attrs)
            embedding = self.caller.embed(item or '[Empty Content]', truncate=True)
            item_embeddings.append(embedding)
        item_embeddings = np.array(item_embeddings)
        np.save(self.embedding_path, item_embeddings)
        pnt(f'embeddings saved to {self.embedding_path}')

    def pca(self):
        pnt('performing PCA')
        pnt('loading embeddings from', self.embedding_path)
        embeddings = np.load(self.embedding_path)
        embed_dim = self.conf.pca

        from sklearn.decomposition import PCA
        pca = PCA(n_components=embed_dim)
        pca.fit(embeddings)
        embeddings = pca.transform(embeddings)

        np.save(self.embedding_path.replace('.npy', f'-pca{embed_dim}.npy'), embeddings)

    def run(self):
        if self.conf.only_pca:
            pnt('skip embedding')
        else:
            self.embed()
        if self.conf.pca:
            self.pca()


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            pca=False,
            only_pca=False,
            seq=False,
            gpu=None,
            tuner=None,
            attrs=None,
            dim=None,
        ),
        makedirs=[]
    ).parse()

    embedder = Embedder(configuration)
    embedder.run()
