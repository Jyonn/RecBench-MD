import abc
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from utils.prompt import SIMPLE_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel


class E5Model(BaseModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = AutoModel.from_pretrained(self.key)
        self.tokenizer = AutoTokenizer.from_pretrained(self.key)
        self.max_len = self.model.config.max_position_embeddings

        self.yes_token = self.tokenizer.convert_tokens_to_ids('yes')
        self.no_token = self.tokenizer.convert_tokens_to_ids('no')

    def ask(self, content) -> Optional[float]:
        raise NotImplementedError("E5 does not support ask method")

    def embed(self, content) -> Optional[torch.Tensor]:
        inputs = self.tokenizer([content], max_length=self.max_len, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        mask = inputs['attention_mask']

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state
            embeddings = embeddings.masked_fill(~mask[..., None].bool(), 0.0)
            embeddings = embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

        return embeddings[0].cpu().detach().numpy()


class E5BaseModel(E5Model):
    pass


class E5LargeModel(E5Model):
    pass
