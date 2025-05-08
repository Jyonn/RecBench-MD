import json

import numpy as np
import pigmento
from pigmento import pnt
from sklearn.manifold import TSNE
from unitok import PickleHandler

from utils.config_init import ConfigInit


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            seq=True,
        ),
        makedirs=[]
    ).parse()

    data = configuration.data.lower()
    model = configuration.model.lower()
    type_ = '.seq' if configuration.seq else ''

    embed_path = f'../BenchLLM4RS/export/{data}/{model}-embeds{type_}.npy'
    embeddings = np.load(embed_path)

    # Assume your data is loaded into a NumPy array called `data`
    # with shape (#items, #dim). For example:
    # data = np.load('your_data.npy')

    # Initialize TSNE to reduce data to 2 dimensions.
    # You can adjust parameters like `perplexity` and `learning_rate` as needed.
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform your data.
    embed_2d = tsne.fit_transform(embeddings)

    PickleHandler.save(embed_2d, embed_path.replace('.npy', '.tsne'))
