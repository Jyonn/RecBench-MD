import heapq
from typing import List, Tuple

import torch
from torch import nn

from loader.code_map import SeqCodeMap as Map
from model.base_discrete_code_model import BaseDiscreteCodeModel
from utils.prompt import SIMPLE_SEQ_SYSTEM


class BaseSeqModel(BaseDiscreteCodeModel):
    PREFIX_PROMPT = SIMPLE_SEQ_SYSTEM

    def __init__(self, code_list: list[int], **kwargs):
        super().__init__(**kwargs)

        current_index = 0
        self.code_list = []
        for number in code_list:
            self.code_list.append(slice(current_index, current_index + number))
            current_index += number
        print(self.code_list)

        self.code_tree = None

    def set_code_tree(self, code_tree):
        self.code_tree = code_tree

    def _get_logits(self, batch):
        embeddings = self.embedding_layer(batch)
        input_embeddings = embeddings['input_embeddings']
        attention_mask = embeddings['attention_mask']

        output = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        states = output.hidden_states[-1]  # [B, L, D]

        return embeddings, self.embedding_layer.classify(states)  # [B, L, C]

    def finetune(self, batch):
        output, logits = self._get_logits(batch)
        cod_input = output['cod_input']  # [B, L]
        cod_mask = output['cod_mask']  # [B, L]

        cod_input = torch.roll(cod_input, -1, 1)  # [B, L]
        cod_mask = torch.roll(cod_mask, -1, 1)  # [B, L]
        cod_labels = torch.ones(cod_input.shape, dtype=torch.long, device=self.device) * -100
        cod_labels[cod_mask] = cod_input[cod_mask]

        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), cod_labels.view(-1))

    def evaluate(self, batch):
        return self.finetune(batch)

    @staticmethod
    def _repeat_tensor(tensor, width):
        b = tensor.size(0)
        return tensor.repeat(width, 1).view(b * width, *tensor.shape[1:])

    def decode(self, batch, width=3, prod_mode=True):
        batch_size_ = batch_size = batch[Map.LEN_COL].size(0)
        batch[Map.BTH_COL] = torch.arange(batch_size)

        # copy batch to batch x k
        for key in batch:
            batch[key] = self._repeat_tensor(batch[key], width)

        batch[Map.LID_COL] = torch.arange(width).repeat_interleave(batch_size)
        batch_size *= width
        beam_start = batch[Map.SOB_COL].to(self.device).unsqueeze(-1)
        beam_length = batch[Map.LOB_COL].to(self.device)
        max_beam_length = beam_length.max().item()

        range_tensor = torch.arange(max_beam_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        ground_truth_indices = beam_start + range_tensor
        input_ids = batch[Map.IPT_COL].to(self.device)
        ground_truth = input_ids[torch.arange(batch_size).unsqueeze(-1), ground_truth_indices][:batch_size_]  # [B, L_]

        last_beams = [[(0, [])]] * batch_size_

        for i in range(max_beam_length):
            current_index = beam_start + i - 1  # type: torch.Tensor
            _, logits = self._get_logits(batch)
            # select [B, C] from logits based on current_index
            logits = logits[torch.arange(batch_size).unsqueeze(-1), current_index]  # [B, 1, C]
            logits = logits.squeeze(1)  # [B, C]
            # only indices in self.code_list[i] is valid, we first mask out other indices
            logit_mask = torch.full_like(logits, -float('inf'))
            logit_mask[:, self.code_list[i]] = logits[:, self.code_list[i]]
            # use softmax to normalize logits
            scores = nn.functional.softmax(logit_mask, dim=-1)  # [B, C]

            # # select top k indices
            # _, indices = scores.topk(width, dim=-1)  # [B, K]
            # # get scores of top k indices
            # scores = scores[torch.arange(batch_size).unsqueeze(-1), indices]  # [B, K]
            valid_indices = self.code_list[i].stop - self.code_list[i].start
            indices = scores.argsort(dim=-1, descending=True)[:, :valid_indices]  # [B, C]
            scores = scores[torch.arange(batch_size).unsqueeze(-1), indices]  # [B, C]

            current_beams = [[] for _ in range(batch_size_)]
            input_ids = batch[Map.IPT_COL].to(self.device)  # [B, L]
            # input_ids = batch[Map.IPT_COL].view(width, batch_size, -1)  # [K, B, L]
            # transpose input_ids to [B, K, L]
            # input_ids = input_ids.transpose(0, 1).contiguous()  # [B, K, L]

            for j in range(batch_size_):
                last_beam = last_beams[j]  # type: List[Tuple[float, List[int]]]

                range_ = width if i > 0 else 1
                for k in range(range_):
                    current_indices = indices[k * batch_size_ + j]
                    current_scores = scores[k * batch_size_ + j]

                    current_num = 0
                    current_path = last_beam[k][1]
                    current_node = self.code_tree
                    for index in current_path:
                        current_node = current_node[index]
                    current_node_set = set(current_node.keys())

                    for score, index in zip(current_scores, current_indices):
                        if current_num >= width:
                            break
                        if index.item() not in current_node_set:
                            continue
                        current_num += 1

                        element = (last_beam[k][0] + score.item(), last_beam[k][1] + [index.item()])

                        # use heapq to maintain top k elements
                        heapq.heappush(current_beams[j], element)
                        if len(current_beams[j]) > width:
                            heapq.heappop(current_beams[j])

            last_beams = current_beams

            replace_values = []
            for k in range(width):
                for j in range(batch_size_):
                    if i == max_beam_length - 1:
                        beam = sorted(current_beams[j], key=lambda x: x[0], reverse=True)[0]
                    else:
                        beam = current_beams[j][k]
                    replace_values.append(beam[1])

            replace_values = torch.tensor(replace_values).to(self.device)
            range_tensor = torch.arange(i + 1).unsqueeze(0).expand(batch_size, -1).to(self.device)
            replace_indices = beam_start + range_tensor
            input_ids[torch.arange(batch_size).unsqueeze(-1), replace_indices] = replace_values

        if not prod_mode:
            # traditional beam search based retrieval mode
            ranks = []
            ground_truth = ground_truth.tolist()
            ground_truth = ['-'.join([str(x) for x in g]) for g in ground_truth]

            for j in range(batch_size_):
                # print()
                # print(j, ground_truth[j])
                beam = sorted(last_beams[j], key=lambda x: x[0], reverse=True)
                candidates = [beam[k][1] for k in range(width)]
                candidates = ['-'.join([str(x) for x in c]) for c in candidates]
                for k, candidate in enumerate(candidates):
                    # print(candidate)
                    if candidate == ground_truth[j]:
                        ranks.append(k + 1)
                        break
                else:
                    ranks.append(-1)
            # print(ranks)
            # exit(0)
            return ranks

        _, logits = self._get_logits(batch)  # [B, L, C]
        scores = nn.functional.softmax(logits, dim=-1)[:batch_size_]  # [B_, L, C]
        range_tensor = torch.arange(max_beam_length).unsqueeze(0).expand(batch_size_, -1).to(self.device)
        indices = beam_start[:batch_size_] + range_tensor
        scores = scores[torch.arange(batch_size_).unsqueeze(-1), indices]  # [B_, L_, C]  # [B, 4, C]
        # get top K indices
        # _, indices = scores.topk(50, dim=-1)  # [B_, L_, K]
        # print(indices)
        # return indices
        ranks = torch.argsort(scores, dim=-1, descending=True).argsort(dim=-1)
        return ranks[torch.arange(batch_size_).unsqueeze(1), torch.arange(max_beam_length).unsqueeze(0), ground_truth]

        # return scores, rank + 1

    def get_special_tokens(self):
        line = self.generate_simple_input_ids('\n')
        numbers = {i: self.generate_simple_input_ids(f'({i}) ') for i in range(1, 128)}
        user = self.generate_simple_input_ids('User behavior sequence: \n')
        item = self.generate_simple_input_ids('Next item: ')
        prefix = self.generate_simple_input_ids(self.PREFIX_PROMPT)

        return line, numbers, user, item, prefix
