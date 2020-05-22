# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch
import torch.nn as nn
from typing import List
from config import Constants
from transformers import BertTokenizer, BertModel


class TeacherSegmentor(nn.Module):
    def __init__(self, device):
        super(TeacherSegmentor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

        self.cnn = nn.Sequential(
            nn.Conv1d(768 + 15, 768 + 15, 7, stride=1, padding=3),
            nn.ReLU()
        )
        self.dropout1 = nn.Dropout(0.1)
        self.cls = nn.Linear(768+15, 2)
        self.device = device
        self.__init_para()

    def forward(self, insts: List[str], golds, dict_datas:torch.Tensor):
        batch_size, seq_len = golds.shape
        insts, mask = self.__tokenize(batch_size, seq_len, insts, golds)
        hidden_state = self.bert_model(insts, attention_mask=mask)[0][:, 1:seq_len+1, :]  # (batch_size, seq_len, hidden_size)
        hidden_state = torch.cat([hidden_state, dict_datas], 2)

        cnn_out = self.cnn(self.dropout1(hidden_state.permute(0, 2, 1)))  # (batch_size, 768+10, seq_len)
        pred = self.cls(cnn_out.permute(0, 2, 1))
        return pred  # (batch_size, seq_len, 2)

    def __tokenize(self, batch_size: int, seq_len: int, insts: List[str], golds: torch.Tensor):
        insts = [self.tokenizer.encode(inst) + [self.tokenizer.pad_token_id] * (seq_len-(len(inst)+1)//2) for inst in insts]
        insts = torch.tensor(insts).to(self.device)
        assert insts.shape[0] == batch_size and insts.shape[1] == seq_len + 2, 'insts tokenizing goes wrong.'
        mask = torch.ones((batch_size, seq_len + 2), dtype=torch.long).to(self.device)
        mask[:, 2:] = golds != Constants.actionPadId
        return insts, mask

    def __init_para(self):
        nn.init.xavier_uniform_(self.cls.weight)
        nn.init.uniform_(self.cls.bias)

