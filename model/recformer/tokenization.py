import torch
from transformers import LongformerTokenizer


class RecformerTokenizer(LongformerTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cls.config = kwargs['config']
        return super().from_pretrained(pretrained_model_name_or_path)

    def item_tokenize(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def encode_item(self, item):

        input_ids = []
        token_type_ids = []
        item = list(item.items())[:self.config.max_attr_num]  # truncate attribute number

        for attribute in item:
            attr_name, attr_value = attribute

            name_tokens = self.item_tokenize(attr_name)
            value_tokens = self.item_tokenize(attr_value)

            attr_tokens = name_tokens + value_tokens
            attr_tokens = attr_tokens[:self.config.max_attr_length]

            input_ids += attr_tokens

            attr_type_ids = [1] * len(name_tokens)
            attr_type_ids += [2] * len(value_tokens)
            attr_type_ids = attr_type_ids[:self.config.max_attr_length]
            token_type_ids += attr_type_ids

        return input_ids, token_type_ids

    def encode_items(self, items):
        """
        Encode a sequence of items.
        the order of items:  [past...present]
        return: [present...past]
        """
        items = items[::-1]  # reverse items order
        items = items[:self.config.max_item_embeddings - 1]  # truncate the number of items, -1 for <s>

        input_ids = [self.bos_token_id]
        item_position_ids = [0]
        token_type_ids = [0]

        for item_idx, item in enumerate(items):
            item_input_ids, item_token_type_ids = self.encode_item(item)
            input_ids += item_input_ids
            token_type_ids += item_token_type_ids

            item_position_ids += [item_idx + 1] * len(item_input_ids)  # item_idx + 1 make idx starts from 1 (0 for <s>)

        input_ids = input_ids[:self.config.max_token_num]
        item_position_ids = item_position_ids[:self.config.max_token_num]
        token_type_ids = token_type_ids[:self.config.max_token_num]

        attention_mask = [1] * len(input_ids)
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1

        inputs = dict(
            input_ids=input_ids,
            item_position_ids=item_position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )

        for k, v in inputs.items():
            inputs[k] = torch.LongTensor([v])

        return inputs
