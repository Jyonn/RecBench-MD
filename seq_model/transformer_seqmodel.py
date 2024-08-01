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

