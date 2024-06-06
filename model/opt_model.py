import abc

from transformers import OPTForCausalLM, AutoTokenizer

from config import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel


class OPTModel(BaseModel, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # use large size opt model
        self.model = OPTForCausalLM.from_pretrained(self.key)  # type: OPTForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.key)

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')

        self.model.to(self.device)

    def generate_input_ids(self, content):
        return self.tokenizer.encode(CHAT_SYSTEM + content + SIMPLE_SUFFIX, return_tensors='pt')


class OPT1BModel(OPTModel):
    pass


class OPT350MModel(OPTModel):
    pass
