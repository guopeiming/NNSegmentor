# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch
import torch.nn as nn
from typing import List
from config import Constants
from transformers import BertTokenizer, BertModel
from model.StackLSTMCell import StackLSTMCell
from model.SubwordLSTMCell import SubwordLSTMCell


class BertStackSegmentor(nn.Module):
    def __init__(self, device):
        super(BertStackSegmentor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=True)
        self.subwStackLSTM = SubwordLSTMCell(768*2, 768, device)
        self.wordStackLSTM = StackLSTMCell(2 * 768, 768, device)
        self.cls = nn.Linear(768*3, 2)
        self.device = device
        self.subword_action_map = torch.tensor([1, 2, 2]).to(self.device)
        self.word_action_map = torch.tensor([0, 1, 1]).to(self.device)
        self.__init_para()

    def forward(self, insts, golds):
        batch_size, seq_len = golds.shape
        insts, mask = self.__tokenize(batch_size, seq_len, insts, golds)
        hidden_state = self.bert_model(insts, attention_mask=mask)[0][:, 1:seq_len + 1, :]  # (batch_size, seq_len, hidden_size)
        lstm_output, _ = self.lstm(hidden_state)  # (batch_size, seq_len, 768*2)

        self.subwStackLSTM.init_stack(2 * seq_len + 2, batch_size)
        self.wordStackLSTM.init_stack(seq_len + 1, batch_size)

        pred = [torch.tensor([[[-1., 1.]]]).expand((batch_size, 1, 2)).to(self.device)]
        for idx in range(1, seq_len, 1):
            subword = self.subwStackLSTM(lstm_output[:, idx - 1, :])  # (batch_size, 768*2)
            word_repre, _ = self.wordStackLSTM(subword)  # (batch_size, 768)
            output = self.cls(torch.cat([word_repre, lstm_output[:, idx, :]], 1))  # (batch_size, 2)

            if self.training:
                subwordOP = self.subword_action_map.index_select(0, golds[:, idx])
                wordOP = self.word_action_map.index_select(0, golds[:, idx])
            else:
                subwordOP = self.subword_action_map.index_select(0, torch.argmax(output, 1))
                wordOP = self.word_action_map.index_select(0, torch.argmax(output, 1))
            self.subwStackLSTM.update_pos(subwordOP)
            self.wordStackLSTM.update_pos(wordOP)

            pred.append(output.unsqueeze(1))
        return torch.cat(pred, 1)  # (batch_size, seq_len, 2)

    def __tokenize(self, batch_size: int, seq_len: int, insts: List[str], golds: torch.Tensor):
        insts = [self.tokenizer.encode(inst) + [self.tokenizer.pad_token_id] * (seq_len - (len(inst) + 1) // 2) for inst
                 in insts]
        insts = torch.tensor(insts).to(self.device)
        assert insts.shape[0] == batch_size and insts.shape[1] == seq_len + 2, 'insts tokenizing goes wrong.'
        mask = torch.ones((batch_size, seq_len + 2), dtype=torch.long).to(self.device)
        mask[:, 2:] = golds != Constants.actionPadId
        return insts, mask

    def __init_para(self):
        nn.init.xavier_uniform_(self.cls.weight)
        nn.init.uniform_(self.cls.bias)

