from model.transformer_model import TransformerModel
from seq_model.base_seqmodel import BaseSeqModel


class TransformerSeqModel(BaseSeqModel, TransformerModel):
    def post_init(self):
        super().post_init()

        self.embedding_layer.llm_embeddings.weight.requires_grad = True

    def get_special_tokens(self):
        line = []
        numbers = {i: [] for i in range(1, 128)}
        user = []
        item = []
        prefix = []

        return line, numbers, user, item, prefix


class Transformer6LSeqModel(TransformerSeqModel):
    N_LAYERS = 6


class Transformer1LSeqModel(TransformerSeqModel):
    N_LAYERS = 1


class Transformer12LSeqModel(TransformerSeqModel):
    N_LAYERS = 12


class Transformer12L768DSeqModel(TransformerSeqModel):
    N_LAYERS = 12
    N_EMBEDDINGS = 768
    N_HEADS = 12


class Transformer24L2048DSeqModel(TransformerSeqModel):
    N_LAYERS = 24
    N_EMBEDDINGS = 2048
    N_HEADS = 16
