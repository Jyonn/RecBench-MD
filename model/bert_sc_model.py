from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.bert_model import BertModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class BertDiscreteCodeModel(BaseDiscreteCodeModel, BertModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 32


class BertBaseSCModel(BertDiscreteCodeModel):
    KEY = 'bert-base-uncased'
