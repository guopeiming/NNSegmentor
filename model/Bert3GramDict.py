# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch
import random
import torch.nn as nn
from typing import List
import torch.nn.init as init
from config import Constants
from transformers import BertTokenizer, BertModel
from torch.nn.modules.transformer import TransformerEncoderLayer


class Bert3GramDict(nn.Module):
    def __init__(self, device, cache_3gram_path):
        super(Bert3GramDict, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

        self.cnn = nn.Sequential(
            nn.Conv1d(768+15, 768+15, 7, stride=1, padding=3),
            nn.ReLU()
        )
        self.dropout1 = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(3*768, 768),
            nn.ReLU()
        )
        self.cls = nn.Linear(768+15, 2)

        self.__3gram_cache = torch.load(cache_3gram_path)

        self.device = device
        self.__init_para()

    def forward(self, insts: List[str], golds: torch.Tensor, dict_datas:torch.Tensor):
        batch_size, seq_len = golds.shape
        insts, insts_seq_len = self.__tokenize(insts)
        bert_output = self.bert_model(insts)[0][:, 1:4, :].view(-1, 3*768)  # (sum(insts_seq_len), 5*768)
        assert bert_output.shape[0] == sum(insts_seq_len), 'bert forward goes wrong.'
        mlp_output = self.mlp(bert_output)  # (sum(insts_seq_len), 768)
        bert_output = torch.split(mlp_output, insts_seq_len, 0)
        bert_output = [torch.cat([inst, torch.zeros((seq_len-inst.shape[0], 768)).to(self.device)], 0).unsqueeze(0) for inst in bert_output]
        bert_output = torch.cat(bert_output, 0)  # (batch_size, seq_len, 768)
        assert bert_output.shape[0] == batch_size and bert_output.shape[1] == seq_len and \
            bert_output.shape[2] == 768, 'bert_out reshape goes wrong.'

        bert_output = torch.cat([bert_output, dict_datas], 2)  # (batch_size, seq_len, 768+10)
        cnn_output = self.cnn(self.dropout1(bert_output.permute(0, 2, 1)))  # (batch_size, 768+10, seq_len)
        pred = self.cls(cnn_output.permute(0, 2, 1))  # (batch_size, seq_len, 2)
        return pred

    def __tokenize(self, insts: List[str]) -> (torch.Tensor, List[int]):
        insts_seq_len = [(len(inst) + 1) // 2 for inst in insts]
        pad_id, cls_id, sep_id, mask_id = self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id
        insts = [[pad_id] + self.tokenizer.encode(inst)[1:(len(inst) + 1) // 2 + 1] + [pad_id] for inst in insts]
        insts_3gram = []
        for inst_seq_len, inst in zip(insts_seq_len, insts):
            for i in range(inst_seq_len):
                if self.training:
                    random_num = random.uniform(0, 1)
                    inst_3gram = inst[i:i + 3] if random_num < 0.5 else [mask_id, inst[i + 1], mask_id]
                else:
                    inst_3gram_str = self.tokenizer.decode(inst[i:i + 3], clean_up_tokenization_spaces=False).split(' ')
                    if i == 0: inst_3gram_str[0] = '-'
                    if i == inst_seq_len - 1: inst_3gram_str[2] = '-'
                    inst_3gram_str = ''.join(inst_3gram_str)
                    inst_3gram = inst[i:i + 3] if inst_3gram_str in self.__3gram_cache else [mask_id, inst[i + 1],
                                                                                             mask_id]
                insts_3gram.append([cls_id] + inst_3gram + [sep_id])
        insts = torch.tensor(insts_3gram).to(self.device)
        assert insts.shape[0] == sum(insts_seq_len) and insts.shape[1] == 5, 'insts tokenizing goes wrong.'
        return insts, insts_seq_len

    def __init_para(self):
        init.xavier_uniform_(self.mlp[0].weight)
        init.uniform_(self.mlp[0].bias)
        init.xavier_uniform_(self.cls.weight)
        init.uniform_(self.cls.bias)

    def pack_state_dict(self):
        res = {'dropo': self.dropout1.state_dict(),
               'bert_model': self.bert_model.state_dict(),
               'cnn': self.cnn.state_dict(),
               'mlp': self.mlp.state_dict(),
               'cls': self.cls.state_dict()}
        return res

