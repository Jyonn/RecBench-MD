import heapq
from distutils.dep_util import newer
from typing import List, Tuple

import torch
from torch import nn

from loader.code_map import SeqCodeMap as Map
from model.base_discrete_code_model import BaseDiscreteCodeModel
from utils.prompt import SIMPLE_SEQ_SYSTEM


class BaseSeqModel(BaseDiscreteCodeModel):
    PREFIX_PROMPT = SIMPLE_SEQ_SYSTEM

    PRED_ALL = True

    def __init__(self, code_list: list[int], **kwargs):
        super().__init__(**kwargs)

        current_index = 0
        self.code_list = []
        self.valid_counts = []
        for number in code_list:
            self.code_list.append(slice(current_index, current_index + number))
            self.valid_counts.append(number)
            current_index += number
        print(self.code_list)

        self.code_tree = None
        self.code_map = None

    def set_code_meta(self, code_tree, code_map):
        self.code_tree = code_tree
        self.code_map = code_map

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

    def finetune(self, batch, **kwargs):
        output, logits = self._get_logits(batch)
        cod_input = output['cod_input']  # [B, L]
        cod_mask = output['cod_mask']  # [B, L]

        if not self.PRED_ALL:
            # 初始化一个全是 -100 的张量，表示被忽略的位置
            cod_labels_last = torch.full((cod_input.size(0),), -100, dtype=torch.long, device=self.device)

            for i in range(cod_input.size(0)):
                # 找出该序列中所有 mask 为 True 的位置
                true_indices = torch.where(cod_mask[i])[0]
                if true_indices.numel() > 0:
                    # 取最后一个位置的索引
                    last_idx = true_indices[-1].item()
                    cod_labels_last[i] = cod_input[i, last_idx]

            # 计算损失
            return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), cod_labels_last)

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

    def prod_decode(self, batch):
        _, logits = self._get_logits(batch)
        # logits: [B, BIGL, C]
        batch_size = logits.size(0)

        decode_start = batch[Map.SOB_COL].to(self.device)  # [B]
        decode_length = batch[Map.LOB_COL].to(self.device)  # [B]
        if decode_length.max().item() != decode_length.min().item():
            raise ValueError('decode length should be the same')

        decode_length = decode_length.max().item()
        bz_ar = torch.arange(batch_size).to(self.device)
        dl_ar = torch.arange(decode_length).to(self.device)

        ground_truth_indices = decode_start.unsqueeze(-1) + dl_ar
        input_ids = batch[Map.IPT_COL].to(self.device)
        ground_truth = input_ids[bz_ar.unsqueeze(-1), ground_truth_indices]  # [B, L]
        logits = logits[bz_ar.unsqueeze(-1), ground_truth_indices]  # [B, L, C]

        # for each position, get index of the ground_truth[b][i] in logits[b][i]
        argsort = logits.argsort(dim=-1, descending=True)  # [B, L, C]
        argsort = argsort.argsort(dim=-1)  # [B, L, C]
        rank = argsort[bz_ar.unsqueeze(-1), dl_ar.unsqueeze(0), ground_truth]  # [B, L]
        return rank

    def decode(self, batch, width=3, decode_mode='hard'):
        # 以下代码经由ChatGPT注释，我忘了啥意思了
        # decode_mode 决定解码方式：'easy' 使用 code map 解码，'hard' 使用 code tree 解码
        # 如果解码模式为 'prod'，则调用 prod_decode 方法
        if decode_mode == 'prod':
            return self.prod_decode(batch)

        easy_decode = decode_mode == 'easy'

        # 获取批次大小
        orig_batch_size = batch[Map.LEN_COL].size(0)
        batch[Map.BTH_COL] = torch.arange(orig_batch_size, device=self.device)  # 为每个样本分配唯一的批次编号
        beam_lengths = batch[Map.LOB_COL].cpu().tolist()

        # 将批次数据扩展为 batch_size x width
        for key in batch:
            batch[key] = self._repeat_tensor(batch[key], width)
        total_batch_size = orig_batch_size * width

        # 扩展输入 ID（每个样本会被重复 width 次）
        batch[Map.LID_COL] = torch.arange(width, device=self.device).repeat_interleave(total_batch_size)
        beam_start = batch[Map.SOB_COL].to(self.device).unsqueeze(-1)  # 每个 beam 的起始位置
        beam_length = batch[Map.LOB_COL].to(self.device)  # 每个 beam 的最大长度
        max_decode_steps = int(beam_length.max().item())  # 获取最大解码长度

        # 构建一个从起始位置到最大解码长度的索引范围
        range_tensor = torch.arange(max_decode_steps, device=self.device).unsqueeze(0).expand(total_batch_size, -1)
        ground_truth_indices = beam_start + range_tensor

        # 获取输入 ID 和真实标签
        input_ids = batch[Map.IPT_COL].to(self.device)
        ground_truth = input_ids[torch.arange(total_batch_size, device=self.device).unsqueeze(-1), ground_truth_indices][:orig_batch_size]

        # 初始化 Beam Search 的 last_beams，每个元素保存 (score, path)
        # last_beams = [[(0, [])]] * orig_batch_size
        last_beams = [[(0.0, [])] for _ in range(orig_batch_size)]
        total_indices = torch.arange(total_batch_size, device=self.device).unsqueeze(-1)

        for step in range(max_decode_steps):
            # 获取当前解码步的 logits
            current_index = beam_start + step - 1
            _, logits = self._get_logits(batch)

            # 根据当前解码步获取 logits
            # logits = logits[torch.arange(total_batch_size).unsqueeze(-1), current_index]
            logits = logits[total_indices, current_index].squeeze(1)

            # 使用掩码屏蔽掉无效的词汇项
            logit_mask = torch.full_like(logits, -float('inf'))
            logit_mask[:, self.code_list[step]] = logits[:, self.code_list[step]]

            # 通过 softmax 转换 logits 为概率分布
            scores = nn.functional.softmax(logit_mask, dim=-1)

            candidate_k = width if easy_decode else self.valid_counts[step]
            topk_scores, topk_indices = scores.topk(candidate_k, dim=-1)

            # 为每个样本初始化一个新的 beam 列表
            new_beams = [[] for _ in range(orig_batch_size)]
            for sample_id in range(orig_batch_size):

                if step >= beam_lengths[sample_id]:
                    # 如果已经到达最大解码长度，直接跳过
                    new_beams[sample_id] = last_beams[sample_id]
                    continue

                # 根据是否为第一步，控制遍历范围
                # range_ = width if step > 0 else 1
                # for beam_idx in range(range_):
                for beam_idx, (cur_score, cur_path) in enumerate(last_beams[sample_id]):
                    # current_indices = indices[beam_idx * orig_batch_size + sample_id]
                    # current_scores = scores[beam_idx * orig_batch_size + sample_id]

                    # current_path = last_beam[beam_idx][1]

                    # 如果是 hard_decode 模式，构建当前节点集合（即决策树的子节点）
                    if easy_decode:
                        # current_node_set = set()  # easy_decode 模式下无需使用 code_map
                        valid_set = None
                    else:
                        # current_node = self.code_tree
                        # for index in current_path:
                        #     current_node = current_node[index]
                        # current_node_set = set(current_node.keys())
                        current_node = self.code_tree
                        for token in cur_path:
                            current_node = current_node.get(token, {})
                        valid_set = set(current_node.keys())

                    global_idx = beam_idx * orig_batch_size + sample_id
                    # 遍历当前候选的 (score, index)，并将满足条件的路径添加到 beam 中
                    # for token_score, token in zip(current_scores, current_indices):
                    for token_score, token in zip(topk_scores[global_idx], topk_indices[global_idx]):
                        token = token.item()

                        # if not easy_decode and token not in current_node_set:
                        if valid_set is not None and token not in valid_set:
                            continue

                        # 更新当前 beam 状态，保持 top-k 路径
                        new_score = cur_score + token_score.item()
                        new_path = cur_path + [token]
                        heapq.heappush(new_beams[sample_id], (new_score, new_path))
                        # element = (last_beam[beam_idx][0] + token_score.item(), last_beam[beam_idx][1] + [token.item()])
                        # heapq.heappush(new_beams[sample_id], element)
                        if len(new_beams[sample_id]) > width:
                            heapq.heappop(new_beams[sample_id])

            # 更新 last_beams 为当前的 beam 状态
            last_beams = new_beams

            # 选择最优路径，更新每个样本的 input_ids
            updated_paths = []
            for beam_idx in range(width):
                for sample_id in range(orig_batch_size):
                    # if i == max_beam_length - 1:
                    if step == beam_lengths[sample_id] - 1:
                        # 如果已经到达最大解码长度，选择最优路径
                        beam = sorted(new_beams[sample_id], key=lambda x: x[0], reverse=True)[0]
                    elif step > beam_lengths[sample_id] - 1:
                        beam = sorted(new_beams[sample_id], key=lambda x: x[0], reverse=True)[0]
                        beam[1].extend([0] * (step + 1 - len(beam[1])))
                    else:
                        # 否则选择当前 top-k 路径
                        beam = new_beams[sample_id][beam_idx]
                    updated_paths.append(beam[1])

            # 更新每个样本的输入 ID
            # updated_paths = torch.tensor(updated_paths).to(self.device)
            # range_tensor = torch.arange(step + 1).unsqueeze(0).expand(total_batch_size, -1).to(self.device)
            # replace_indices = beam_start + range_tensor
            # input_ids[torch.arange(total_batch_size).unsqueeze(-1), replace_indices] = updated_paths
            replace_tensor = torch.tensor(updated_paths, device=self.device)
            step_range = torch.arange(step + 1, device=self.device).unsqueeze(0).expand(total_batch_size, -1)
            indices_update = beam_start + step_range
            input_ids[torch.arange(total_batch_size, device=self.device).unsqueeze(-1), indices_update] = replace_tensor

        # 计算每个样本的排名（基于 ground_truth）
        ranks = []
        # ground_truth = ground_truth.tolist()
        # ground_truth = ['-'.join([str(x) for x in g]) for g in ground_truth]
        #
        # for sample_id in range(orig_batch_size):
        #     length = beam_length_[sample_id]
        #     beam = sorted(last_beams[sample_id], key=lambda x: x[0], reverse=True)
        #     candidates = [beam[k][1] for k in range(width)]
        #     candidates = ['-'.join([str(x) for x in c[:length]]) for c in candidates]
        #     for beam_idx, candidate in enumerate(candidates):
        #         if candidate == ground_truth[sample_id]:
        #             ranks.append(beam_idx + 1)  # 如果候选路径与 ground_truth 相同，记录排名
        #             break
        #     else:
        #         ranks.append(-1)  # 如果没有匹配的路径，返回 -1
        #
        # return ranks  # 返回每个样本的排名
        ground_truth_str = ['-'.join(map(str, gt.tolist())) for gt in ground_truth]
        for sample_id in range(orig_batch_size):
            length = beam_lengths[sample_id]
            candidates = [
                '-'.join(map(str, beam[1][:length]))
                for beam in sorted(last_beams[sample_id], key=lambda x: x[0], reverse=True)
            ]
            for rank, cand in enumerate(candidates, start=1):
                if cand == ground_truth_str[sample_id]:
                    ranks.append(rank)
                    break
            else:
                ranks.append(-1)
        return ranks

    def get_special_tokens(self):
        line = self.generate_simple_input_ids('\n')
        numbers = {i: self.generate_simple_input_ids(f'({i}) ') for i in range(1, 128)}
        user = self.generate_simple_input_ids('User behavior sequence: \n')
        item = self.generate_simple_input_ids('Next item: ')
        prefix = self.generate_simple_input_ids(self.PREFIX_PROMPT)

        return line, numbers, user, item, prefix
