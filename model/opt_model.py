import abc

from pigmento import pnt
from transformers import OPTForCausalLM, AutoTokenizer

from utils.prompt import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel


class OPTModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = CHAT_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # use large size opt model
        self.model = OPTForCausalLM.from_pretrained(self.key, torch_dtype=self.get_dtype())  # type: OPTForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.key)
        self.max_len = self.model.config.max_position_embeddings
        # self.max_len = 1024

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')


class OPT1BModel(OPTModel):
    KEY = 'facebook/opt-1.3b'


class OPT350MModel(OPTModel):
    KEY = 'facebook/opt-350m'
