import abc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompt import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel
from utils.auth import HF_KEY


class LongContextModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = CHAT_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # use large size opt model
        self.model = AutoModelForCausalLM.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY)
        self.max_len = 1e5

        # self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        # self.no_token = self.tokenizer.convert_tokens_to_ids('NO')
        self.yes_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('YES'))[0]
        self.no_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('NO'))[0]

    def _generate_input_ids(self, content):
        return self.tokenizer.encode(content, return_tensors='pt', add_special_tokens=False)


class QWen2TH7BModel(LongContextModel):
    pass


class GLM4TH9BModel(LongContextModel):
    pass


class Mistral7BModel(LongContextModel):
    pass


class Phi3TH7BModel(LongContextModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2e3


class Phi2TH3BModel(LongContextModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2e3
