from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from pigmento import pnt

from loader.map import Map
from utils import model


class BaseModel:
    KEY = None
    PREFIX_PROMPT: str
    SUFFIX_PROMPT: str
    AS_DICT: bool = False
    BIT: int

    def __init__(self, device):
        self.device = device
        self.device_ids = None
        if isinstance(device, tuple):
            self.device, self.device_ids = device

        self.key = model.match(self.get_name()) or self.KEY

        self.is_parallel = False
        self.model = None
        self.tokenizer = None
        self.max_len = None

        self.yes_token = None
        self.no_token = None

        self.use_lora = False

        # self.loss_fct = torch.nn.CrossEntropyLoss()
        self.loss_fct = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=0)
        self.softmax_sft = torch.nn.Softmax()

    def get_dtype(self):
        if self.BIT == 16:
            return torch.bfloat16
        if self.BIT == 32:
            return torch.float32
        raise ValueError(f'unsupported bit: {self.BIT}')

    def post_init(self):
        self.model.to(self.device)
        if self.device_ids is not None:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            self.is_parallel = True
        return self

    @property
    def label_tokens(self):
        return torch.tensor([self.no_token, self.yes_token])

    def prepare_model_finetuning(self, conf, inference_mode=False):
        if not conf.use_lora:
            pnt(f'fully finetuning {self.get_name()} model without lora')
            return
        self.use_lora = True

        pnt(f'finetuning {self.get_name()} model with lora ({conf.lora_r}, {conf.lora_alpha}, {conf.lora_dropout})')
        peft_config = LoraConfig(
            inference_mode=inference_mode,
            r=conf.lora_r,
            lora_alpha=conf.lora_alpha,
            lora_dropout=conf.lora_dropout
        )
        self.model = get_peft_model(self.model, peft_config)

    def save(self, path):
        # torch.save(self.model.state_dict(), path)
        module = self.model
        if self.is_parallel:
            module = self.model.module
        if self.use_lora:
            # only save lora parameters
            state_dict = dict()
            for k, v in module.state_dict().items():
                if 'lora' in k:
                    state_dict[k] = v
        else:
            state_dict = module.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        pnt(f'loading finetuned model from {path}')
        state_dict = torch.load(path, map_location='cpu')
        state_dict_ = dict()
        assert self.is_parallel is False  # it can be true after loading
        for k in state_dict:
            if k.startswith('module.'):
                state_dict_[k[7:]] = state_dict[k]
            else:
                state_dict_[k] = state_dict[k]
        self.model.load_state_dict(state_dict_, strict=False)

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Model', '').lower()

    def generate_input_ids(self, content, wrap_ask=True) -> torch.Tensor:
        # raise NotImplemented
        if wrap_ask:
            content = self.PREFIX_PROMPT + content + self.SUFFIX_PROMPT
        return self.tokenizer.encode(content, return_tensors='pt', add_special_tokens=False)

    def generate_simple_input_ids(self, content) -> list:
        return self.tokenizer.encode(content or '', add_special_tokens=False)

    def _get_scores(self, batch):
        input_ids = batch[Map.IPT_COL].to(self.device)
        length = batch[Map.LEN_COL].to(self.device)
        max_len = input_ids.size(-1)
        attention_mask = torch.arange(max_len).expand(input_ids.size(0), max_len).to(self.device) < length.view(-1, 1)
        logits = self.model(input_ids, attention_mask=attention_mask).logits  # [B, L, V]
        indices = (batch[Map.LEN_COL] - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1)).to(self.device)
        logits = torch.gather(logits, 1, indices).squeeze(1)  # [B, V]
        logits = logits[:, self.label_tokens]  # [B, 2]
        scores = self.softmax_sft(logits)  # [B, 2]
        return scores[:, 1]

    def finetune(self, batch):
        scores = self._get_scores(batch)
        return self.loss_fct(scores.float(), batch[Map.LBL_COL].to(self.device).float())

    def evaluate(self, batch):
        scores = self._get_scores(batch)  # [B, V=30522]
        return scores.detach().cpu().tolist()

    def ask(self, content) -> Optional[float]:
        input_ids = self.generate_input_ids(content, wrap_ask=True)
        input_ids = input_ids.to(self.device)
        input_len = input_ids.size(-1)
        if input_len > self.max_len:
            return

        # feed-forward
        with torch.no_grad():
            output = self.model(input_ids)
            logits = output.logits

        # get logits of last token
        logits = logits[0, -1, :]
        yes_score, no_score = logits[self.yes_token].item(), logits[self.no_token].item()

        # get softmax of [yes, no]
        yes_prob, _ = self.softmax(torch.tensor([yes_score, no_score])).tolist()
        return yes_prob

    def embed(self, content) -> Optional[torch.Tensor]:
        input_ids = self.generate_input_ids(content, wrap_ask=False)
        input_ids = input_ids.to(self.device)

        input_len = input_ids.size(-1)
        if input_len > self.max_len:
            return

        # feed-forward
        with torch.no_grad():
            output = self.model(input_ids, output_hidden_states=True)
        # get embeddings of last token
        embeddings = output.hidden_states[-1][0, -1, :]  # type: torch.Tensor
        embeddings = embeddings.float()
        return embeddings.cpu().detach().numpy()

    def __call__(self, content):
        return self.ask(content)

    def get_special_tokens(self):
        line = self.generate_simple_input_ids('\n')
        numbers = {i: self.generate_simple_input_ids(f'({i}) ') for i in range(1, 128)}
        user = self.generate_simple_input_ids('User behavior sequence: \n')
        item = self.generate_simple_input_ids('Candidate item: ')
        prefix = self.generate_simple_input_ids(self.PREFIX_PROMPT)
        suffix = self.generate_simple_input_ids(self.SUFFIX_PROMPT)

        return line, numbers, user, item, prefix, suffix
