from model.base_dense_code_model import BaseDenseCodeModel
from model.opt_model import OPTModel
from utils.prompt import CHAT_SYSTEM, SIMPLE_SUFFIX


class OPTDenseCodeModel(BaseDenseCodeModel, OPTModel):
    PREFIX_PROMPT = CHAT_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16


class OPT1BDCModel(OPTDenseCodeModel):
    KEY = 'facebook/opt-1.3b'


class OPT350MDCModel(OPTDenseCodeModel):
    KEY = 'facebook/opt-350m'
