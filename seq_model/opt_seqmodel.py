from model.opt_model import OPTModel
from seq_model.base_seqmodel import BaseSeqModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class OPTSeqModel(BaseSeqModel, OPTModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16


class OPT1BSeqModel(OPTSeqModel):
    KEY = 'facebook/opt-1.3b'


class OPT350MSeqModel(OPTSeqModel):
    KEY = 'facebook/opt-350m'
