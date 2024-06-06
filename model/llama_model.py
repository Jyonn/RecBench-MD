import abc

from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer

from config import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel


class LlamaModel(BaseModel, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = LlamaForCausalLM.from_pretrained(self.key)  # type: LlamaForCausalLM
        self.tokenizer = LlamaTokenizer.from_pretrained(self.key)

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')

        self.model.to(self.device)

    def generate_input_ids(self, content) -> float:
        return self.tokenizer.encode(CHAT_SYSTEM + content + SIMPLE_SUFFIX, return_tensors='pt')


class Llama1Model(LlamaModel):
    pass
