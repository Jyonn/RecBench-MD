from transformers import OPTForCausalLM, AutoTokenizer

from config import CHAT_SYSTEM, SIMPLE_SUFFIX
from llm.base_llm import BaseLLM


class OPT(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # use large size opt model
        self.model = OPTForCausalLM.from_pretrained('facebook/opt-1.3b')  # type: OPTForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')

        self.yes_token = self.tokenizer.convert_tokens_to_ids('YES')
        self.no_token = self.tokenizer.convert_tokens_to_ids('NO')

        self.model.to(self.device)

    def generate_input_ids(self, content):
        return self.tokenizer.encode(CHAT_SYSTEM + content + SIMPLE_SUFFIX, return_tensors='pt')
