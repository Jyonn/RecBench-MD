import abc

from transformers import LlamaForCausalLM, LlamaTokenizer

from config import SIMPLE_SUFFIX, SIMPLE_SYSTEM
from model.base_model import BaseModel


class LlamaModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = LlamaForCausalLM.from_pretrained(
            self.key,
            # device_map=
        )  # type: LlamaForCausalLM
        self.tokenizer = LlamaTokenizer.from_pretrained(self.key)
        self.max_len = self.model.config.max_position_embeddings

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')

        self.model.to(self.device)


class Llama1Model(LlamaModel):
    pass


class Llama2Model(LlamaModel):
    pass
