from model.bert_model import BertModel
from model.opt_model import OPTModel
from seq_model.base_seqmodel import BaseSeqModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class BertSeqModel(BaseSeqModel, BertModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 32


class BertBaseSeqModel(BertSeqModel):
    KEY = 'bert-base-uncased'
