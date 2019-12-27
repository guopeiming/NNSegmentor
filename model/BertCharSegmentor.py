# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch
import torch.nn as nn
from typing import List
from config import Constants
from transformers import BertTokenizer, BertModel


class BertCharSegmentor(nn.Module):
    def __init__(self, device):
        super(BertCharSegmentor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

        # self.lstm = nn.LSTM(768, 768, num_layers=1, batch_first=True, bidirectional=True)
        self.cls = nn.Linear(768, 2)
        # self.cls = nn.Linear(768*2, 2)
        self.device = device
        self.__init_para()

    def forward(self, insts: List[str], golds):
        batch_size, seq_len = golds.shape
        insts, mask = self.__tokenize(batch_size, seq_len, insts, golds)
        hidden_state = self.bert_model(insts, attention_mask=mask)[0][:, 1:seq_len+1, :]  # (batch_size, seq_len, hidden_size)
        # lstm_output, _ = self.lstm(hidden_state)  # (batch_size, seq_len, 768*2)

        # pred = self.cls(lstm_output)  # (batch_size, seq_len, 2)
        pred = self.cls(hidden_state)
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

