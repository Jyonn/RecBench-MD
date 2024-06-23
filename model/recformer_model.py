import abc
from typing import Optional

import torch
from transformers import T5Config

from config import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel
from model.p5.modeling_p5 import P5
from model.p5.tokenization import P5Tokenizer
from model.recformer.models import RecformerConfig, Recformer
from model.recformer.tokenization import RecformerTokenizer


class RecformerModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_SUFFIX
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    AS_DICT = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.backbone, self.key = self.key.split('$')

        config = RecformerConfig.from_pretrained(self.backbone)
        config.max_attr_num = 3  # max number of attributes for each item
        config.max_attr_length = 32  # max number of tokens for each attribute
        config.max_item_embeddings = 51  # max number of items in a sequence +1 for cls token
        config.attention_window = [64] * 12  # attention window for each layer

        self.tokenizer = RecformerTokenizer.from_pretrained(self.backbone, config=config)  # type: RecformerTokenizer

        self.model = Recformer(config)
        self.load_state_dict()

    def ask(self, content) -> Optional[float]:
        raise NotImplementedError("Recformer does not support ask method")

    def embed(self, content) -> Optional[torch.Tensor]:
        inputs = self.tokenizer.encode_items(content)

        self.model.eval()
        with torch.no_grad():
            embedding = self.model(
                input_ids=inputs['input_ids'].to(self.device),
                item_position_ids=inputs['item_position_ids'].to(self.device),
                token_type_ids=inputs['token_type_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                global_attention_mask=inputs['global_attention_mask'].to(self.device),
            )

        return embedding.pooler_output[0].cpu().detach().numpy()

    def load_state_dict(self):
        state_dict = torch.load(self.key, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('longformer')}
        state_dict = {k.replace('longformer.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
