# @Author : guopeiming
# @Datetime : 2019/12/09 19:49
# @File : BertWordSegmentor.py
# @Last Modify Time : 2019/12/09 19:49
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch
import torch.nn as nn
from config import Constants
from transformers import BertTokenizer, BertModel


class BertWordSegmentor(nn.Module):
    def __init__(self, device):
        super(BertWordSegmentor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

        self.cls = nn.Linear(768, 2)
        self.device = device
        self.__init_para()

    def forward(self, insts, golds):
        batch_size, seq_len = golds.shape
        insts = [self.tokenizer.encode(inst) + [self.tokenizer.pad_token_id]*(seq_len-(len(inst)+1)//2) for inst in insts]
        insts = torch.tensor(insts).to(self.device)
        assert insts.shape[0] == batch_size and insts.shape[1] == seq_len+2, 'insts tokenizing goes wrong.'
        attention_mask = torch.ones((batch_size, seq_len+2), dtype=torch.long).to(self.device)
        attention_mask[:, 2:] = golds != Constants.actionPadId
        hidden_state = self.bert_model(insts, attention_mask=attention_mask)[0]  # (batch_size, seq_len+2, hidden_size)
        pred = self.cls(hidden_state)  # (batch_size, seq_len+2, 2)
        return pred[:, 1:seq_len+1, :]  # (batch_size, seq_len, 2)

    def __init_para(self):
        nn.init.xavier_uniform_(self.cls.weight)
        nn.init.uniform_(self.cls.bias)

