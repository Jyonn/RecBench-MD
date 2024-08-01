from model.base_discrete_code_model import BaseDiscreteCodeModel
from model.transformer_model import TransformerModel


class TransformerCodeModel(BaseDiscreteCodeModel, TransformerModel):
    def post_init(self):
        super().post_init()

        self.embedding_layer.llm_embeddings.weight.requires_grad = True
