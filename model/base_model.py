import torch

from utils import model


class BaseModel:
    def __init__(self, device):
        self.device = device

        self.key = model.match(self.get_name())

        self.model = None
        self.tokenizer = None

        self.yes_token = None
        self.no_token = None

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Model', '').lower()

    def generate_input_ids(self, content) -> torch.Tensor:
        raise NotImplemented

    def ask(self, content) -> float:
        input_ids = self.generate_input_ids(content)
        input_ids = input_ids.to(self.device)

        # feed-forward
        with torch.no_grad():
            output = self.model(input_ids)
            logits = output.logits

        # get logits of last token
        logits = logits[0, -1, :]
        yes_score, no_score = logits[self.yes_token].item(), logits[self.no_token].item()

        # get softmax of [yes, no]
        softmax = torch.nn.Softmax(dim=0)
        yes_prob, _ = softmax(torch.tensor([yes_score, no_score])).tolist()
        return yes_prob
