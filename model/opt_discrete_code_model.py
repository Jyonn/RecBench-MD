from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.opt_model import OPTModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class OPTDiscreteCodeModel(BaseDiscreteCodeModel, OPTModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16


class OPT1BSCModel(OPTDiscreteCodeModel):
    KEY = 'facebook/opt-1.3b'


class OPT350MSCModel(OPTDiscreteCodeModel):
    KEY = 'facebook/opt-350m'
