import abc

from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM
from model.base_model import BaseModel
from utils.auth import HF_KEY


class LlamaModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16
    NUM_LAYERS = 32

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
            torch_dtype=self.get_dtype(),
            **params,
        )  # type: LlamaForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.key,
            **params,
        )
        # self.max_len = self.model.config.max_position_embeddings
        self.max_len = 1024

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')


class Llama1Model(LlamaModel):
    KEY = 'huggyllama/llama-7b'


class Llama2Model(LlamaModel):
    KEY = 'meta-llama/Llama-2-7b-hf'


class Llama3Model(LlamaModel):
    KEY = 'meta-llama/Meta-Llama-3-8B'


class Llama31Model(LlamaModel):
    KEY = 'meta-llama/Meta-Llama-3.1-8B'
