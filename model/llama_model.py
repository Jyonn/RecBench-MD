import abc

import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM
from model.base_model import BaseModel
from utils.auth import HF_KEY


class LlamaModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        params = dict()
        if not self.key.startswith('/'):
            params = dict(
                trust_remote_code=True,
                token=HF_KEY,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.key,
            torch_dtype=torch.bfloat16,
            **params,
        )  # type: LlamaForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.key,
            **params,
        )
        self.max_len = self.model.config.max_position_embeddings

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')


class Llama1Model(LlamaModel):
    pass


class Llama2Model(LlamaModel):
    pass


class Llama3Model(LlamaModel):
    pass
