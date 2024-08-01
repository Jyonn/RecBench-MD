import torch.nn
from transformers import GPT2Config, GPT2LMHeadModel

from model.base_model import BaseModel


class TransformerModel(BaseModel):
    KEY = ''
    SUFFIX_PROMPT = ''
    PREFIX_PROMPT = ''
    BIT = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert not self.use_lora, 'Transformer model should be trained from scratch, not with LORA'

        # model is an empty transformer model, train from scratch, not a pretrained LM
        config = GPT2Config(
            vocab_size=2,
            n_positions=512,
            n_embd=64,
            n_layer=3,
            n_head=8,
            n_ctx=512,
        )
        self.model = GPT2LMHeadModel(config)

        self.yes_token = 0
        self.no_token = 1

    def generate_input_ids(self, content, wrap_ask=True) -> torch.Tensor:
        raise NotImplementedError('generate_input_ids is not supported for TransformerModel')

    def generate_simple_input_ids(self, content) -> list:
        raise NotImplementedError('generate_simple_input_ids is not supported for TransformerModel')

    def get_special_tokens(self):
        line = []
        numbers = {i: [] for i in range(1, 128)}
        user = []
        item = []
        prefix = []
        suffix = []

        return line, numbers, user, item, prefix, suffix
