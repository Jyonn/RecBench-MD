import abc

import torch

from transformers.models.bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert import BertTokenizer

from config import SIMPLE_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel


class BertModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = BertForMaskedLM.from_pretrained(self.key)  # type: BertForMaskedLM
        self.tokenizer = BertTokenizer.from_pretrained(self.key)  # type: BertTokenizer
        self.max_len = self.model.config.max_position_embeddings

        self.cls_token = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.mask_token = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.yes_token = self.tokenizer.convert_tokens_to_ids('yes')
        self.no_token = self.tokenizer.convert_tokens_to_ids('no')

        # load to device
        self.model.to(self.device)


class BertBaseModel(BertModel):
    pass


class BertLargeModel(BertModel):
    pass
