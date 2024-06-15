import abc

import torch
from transformers import T5Config

from config import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel
from model.p5.modeling_p5 import P5
from model.p5.tokenization import P5Tokenizer


class P5Model(BaseModel, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.backbone = 't5-base'
        self.max_len = 512
        config = T5Config.from_pretrained(self.backbone)

        self.tokenizer = P5Tokenizer.from_pretrained(
            self.backbone,
            max_length=self.max_len,
            do_lower_case=False,
        )

        self.model = P5.from_pretrained(
            self.backbone,
            config=config
        )

        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer

        self.load_state_dict()

    def load_state_dict(self):
        state_dict = torch.load(self.key, map_location='cpu')
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module."):
                new_key = key[len("module."):]
                state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)

    def generate_input_ids(self, content, wrap_ask=True) -> torch.Tensor:
        if wrap_ask:
            content = CHAT_SYSTEM + content + SIMPLE_SUFFIX
        return self.tokenizer.encode(content, return_tensors='pt')


class P5BeautyModel(P5Model):
    pass
