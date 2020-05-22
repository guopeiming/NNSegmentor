import torch.nn as nn
import torch
from config import Constants
from transformers import BertTokenizer, BertModel


class BertMdoel_test(nn.Module):
    def __init__(self, device):
        super(BertMdoel_test, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.device = device

        self.cls = nn.Linear(768, 2)

    def forward(self, insts, golds):
        batch_size, seq_len = golds.shape
        insts = [''.join([ch + ' ' for ch in inst]).strip() for inst in insts]
        insts = [self.tokenizer.encode(inst) + [self.tokenizer.pad_token_id] * (seq_len - (len(inst) + 1) // 2) for inst
                 in insts]
        insts = torch.tensor(insts).to(self.device)
        assert insts.shape[0] == batch_size and insts.shape[1] == seq_len + 2, 'insts tokenizing goes wrong.'
        mask = torch.ones((batch_size, seq_len + 2), dtype=torch.long).to(self.device)
        mask[:, 2:] = golds != Constants.actionPadId

        hidden_state = self.bert(insts, attention_mask=mask)[0][:, 1:seq_len + 1, :]  # (batch_size, seq_len, hidden_size)
        pred = self.cls(hidden_state)
        return pred  # (batch_size, seq_len, 2)


class Bert3Gram_test(nn.Module):
    def __init__(self, device, embeddgins):
        super(Bert3Gram_test, self).__init__()
        self.embeddings = embeddgins
        self.dropout1 = nn.Dropout(0.1)
        self.cls = nn.Linear(768, 2)
        self.cnn = nn.Sequential(
            nn.Conv1d(768, 768, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.device = device

    def forward(self, insts, golds):
        batch_size, seq_len = golds.shape
        insts_tensor = []
        for inst in insts:
            inst = '-' + inst + '-'
            inst_tensor = []
            for i in range(len(inst) - 2):
                if inst[i:i+3] in self.embeddings:
                    inst_tensor.append(self.embeddings[inst[i:i+3]])
                else:
                    inst_tensor.append(torch.zeros((1, 768)))
            inst_tensor = torch.cat(inst_tensor + [torch.zeros((1, 768))]*(seq_len - len(inst)+2), 0).to(self.device)
            insts_tensor.append(inst_tensor.unsqueeze(0))
        insts_tensor = torch.cat(insts_tensor, 0)  # (batch_size, seq_len, 768)
        cnn_output = self.cnn(self.dropout1(insts_tensor.permute(0, 2, 1)))  # (batch_size, 768, seq_len)
        pred = self.cls(cnn_output.permute(0, 2, 1))



        return pred

