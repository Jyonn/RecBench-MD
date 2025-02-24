from model.llama_model import LlamaModel, Llama1Model, Llama3Model
from seq_model.base_seqmodel import BaseSeqModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class LlamaSeqModel(BaseSeqModel, LlamaModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16


class Llama1SeqModel(LlamaSeqModel, Llama1Model):
    pass


class Llama3SeqModel(LlamaSeqModel, Llama3Model):
    pass
