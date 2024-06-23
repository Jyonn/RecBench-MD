import abc

import torch
from transformers import OPTForCausalLM, AutoTokenizer

from config import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel


class OPTModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = CHAT_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # use large size opt model
        self.model = OPTForCausalLM.from_pretrained(self.key, torch_dtype=torch.bfloat16)  # type: OPTForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.key)
        self.max_len = self.model.config.max_position_embeddings

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')


class OPT1BModel(OPTModel):
    pass


class OPT350MModel(OPTModel):
    pass
