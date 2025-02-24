from model.qwen_model import QWenModel, QWen2TH0D5BModel, QWen2TH1D5BModel, QWen2TH7BModel
from seq_model.base_seqmodel import BaseSeqModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class QWenSeqModel(BaseSeqModel, QWenModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16


class QWen2TH0D5BSeqModel(QWenSeqModel, QWen2TH0D5BModel):
    pass


class QWen2TH1D5BSeqModel(QWenSeqModel, QWen2TH1D5BModel):
    pass


class QWen2TH7BSeqModel(QWenSeqModel, QWen2TH7BModel):
    pass
