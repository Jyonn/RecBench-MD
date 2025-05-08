import torch.nn
from transformers.modeling_outputs import CausalLMOutput

from model.base_model import BaseModel


import json

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig


class CaserConfig(BertConfig):
    def __init__(
            self,
            num_vertical,
            num_horizontal,
            max_length,
            dropout=0.1,
            **kwargs,
    ):
        super(CaserConfig, self).__init__(**kwargs)
        self.num_vertical = num_vertical
        self.num_horizontal = num_horizontal
        self.dropout = dropout
        self.max_length = max_length


class CaserBaseModel(nn.Module):
    def __init__(self, config: CaserConfig):
        super(CaserBaseModel, self).__init__()

        self.config = config

        # vertical and horizontal conv
        self.vertical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.num_vertical,
            kernel_size=(config.max_length, 1)
        )
        lengths = [i + 1 for i in range(config.max_length)]
        self.horizontal_conv = nn.ModuleList([nn.Conv2d(
            in_channels=1,
            out_channels=config.num_horizontal,
            kernel_size=(i, config.hidden_size)
        ) for i in lengths])

        self.fc_vertical_size = config.num_vertical * config.hidden_size
        self.fc_horizontal_size = config.num_horizontal * config.max_length
        self.fc = nn.Linear(
            in_features=self.fc_vertical_size + self.fc_horizontal_size,
            out_features=config.hidden_size,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.conv_act = self.fc_act = nn.ReLU()

    def forward(
            self,
            inputs_embeds: torch.Tensor,
            **kwargs,
    ):
        batch_size = inputs_embeds.size(0)
        inputs_embeds = inputs_embeds.unsqueeze(dim=1)

        vertical_output = self.vertical_conv(inputs_embeds)
        vertical_output = F.adaptive_max_pool2d(vertical_output, (1, vertical_output.size(3)))

        vertical_output = vertical_output.view(batch_size, -1)

        horizontal_outputs = []
        for conv in self.horizontal_conv:
            conv_output = self.conv_act(conv(inputs_embeds).squeeze(3))
            pool_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            horizontal_outputs.append(pool_output.view(batch_size, -1))
        horizontal_output = torch.cat(horizontal_outputs, dim=1)

        output = torch.cat([vertical_output, horizontal_output], 1)
        output = self.dropout(output)

        fc_output = self.fc_act(self.fc(output))

        # return fc_output
        return CausalLMOutput(hidden_states=(fc_output,))

    def get_input_embeddings(self):
        return nn.Embedding(1, self.config.hidden_size)


class CaserModel(BaseModel):
    KEY = ''
    SUFFIX_PROMPT = ''
    PREFIX_PROMPT = ''
    BIT = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert not self.use_lora, 'Caser model should be trained from scratch, not with LORA'

        # model is an empty transformer model, train from scratch, not a pretrained LM
        config = CaserConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_vertical=4,
            num_horizontal=16,
            max_length=5,
        )
        self.model = CaserBaseModel(config)

        self.yes_token = 0
        self.no_token = 1

    def generate_input_ids(self, content, wrap_ask=True) -> torch.Tensor:
        raise NotImplementedError('generate_input_ids is not supported for Caser')

    def generate_simple_input_ids(self, content) -> list:
        raise NotImplementedError('generate_simple_input_ids is not supported for Caser')

    def get_special_tokens(self):
        line = []
        numbers = {i: [] for i in range(1, 128)}
        user = []
        item = []
        prefix = []
        suffix = []

        return line, numbers, user, item, prefix, suffix
