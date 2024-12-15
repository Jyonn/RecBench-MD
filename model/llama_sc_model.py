from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.llama_model import LlamaModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class LlamaSCModel(BaseDiscreteCodeModel, LlamaModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16


class Llama3SCModel(LlamaSCModel):
    KEY = 'meta-llama/Meta-Llama-3-8B'
