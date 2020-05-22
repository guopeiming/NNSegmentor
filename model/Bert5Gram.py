# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch
import torch.nn as nn
from typing import List
import torch.nn.init as init
from config import Constants
from transformers import BertTokenizer, BertModel
from torch.nn.modules.transformer import TransformerEncoderLayer


class Bert5Gram(nn.Module):
    def __init__(self, device):
        super(Bert5Gram, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')

        self.cnn = nn.Sequential(
            nn.Conv1d(768, 768, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.dropout1 = nn.Dropout(0.1)
        # self.transformer = TransformerEncoderLayer(768, 12)
        self.mlp = nn.Sequential(
            nn.Linear(5*768, 768),
            nn.ReLU()
        )
        self.cls = nn.Linear(768, 2)

        self.device = device
        self.__init_para()

    def forward(self, insts: List[str], golds: torch.Tensor):
        batch_size, seq_len = golds.shape
        insts, insts_seq_len = self.__tokenize(insts)
        bert_output = self.bert_model(insts)[0][:, 1:6, :].view(-1, 5*768)  # (sum(insts_seq_len), 5*768)
        assert bert_output.shape[0] == sum(insts_seq_len), 'bert forward goes wrong.'
        bert_output = torch.split(bert_output, insts_seq_len, 0)
        bert_output = [torch.cat([inst, torch.zeros((seq_len-inst.shape[0], 5*768)).to(self.device)], 0).unsqueeze(0) for inst in bert_output]
        bert_output = torch.cat(bert_output, 0)  # (batch_size, seq_len, 5*768)
        assert bert_output.shape[0] == batch_size and bert_output.shape[1] == seq_len and \
            bert_output.shape[2] == 5*768, 'bert_out reshape goes wrong.'
        mlp_output = self.mlp(bert_output)  # (batch_size, seq_len, 768)
        # transformer_output = self.transformer(self.dropout1(mlp_output.permute(1, 0, 2)))  # (seq_len, batch_size, 768)
        # pred = self.cls(transformer_output.permute(1, 0, 2))  # (batch_size, seq_len, 2)
        cnn_output = self.cnn(self.dropout1(mlp_output.permute(0, 2, 1)))  # (batch_size, 768, seq_len)
        pred = self.cls(cnn_output.permute(0, 2, 1))  # (batch_size, seq_len, 2)
        return pred

    # def forward(self, insts: List[str], golds: torch.Tensor):
    #     batch_size, seq_len = golds.shape
    #     insts, mask = self.__tokenize(batch_size, seq_len, insts, golds)
    #     hidden_state = self.bert_model(insts, attention_mask=mask)[0][:, 1:seq_len + 1, :]  # (batch_size, seq_len, hidden_size)
    #     # cnn_output = self.cnn(hidden_state.permute(0, 2, 1))  # (batch_size, 768, seq_len)
    #     transformer_output = self.transformer(hidden_state.permute(1, 0, 2))  # (seq_len, batch_size, 768)
    #     # cnn_output = self.cnn(bert_output.permute(0, 2, 1))  # (batch_size, 5*768, seq_len)
    #     mlp_output = self.mlp(transformer_output.permute(1, 0, 2))  # (batch_size, seq_len, 768)
    #     # mlp_output = self.mlp(cnn_output.permute(0, 2, 1))  # (batch_size, seq_len, 768)
    #     pred = self.cls(self.dropout1(mlp_output))  # (batch_size, seq_len, 2)
    #     return pred

    def __tokenize(self, insts: List[str]) -> (torch.Tensor, List[int]):
        insts_seq_len = [(len(inst)+1)//2 for inst in insts]
        pad_id, cls_id, sep_id = self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        insts = [[pad_id]*2 + self.tokenizer.encode(inst)[1:(len(inst)+1)//2+1] + [pad_id]*2 for inst in insts]
        insts_5gram = []
        for seq_len, inst in zip(insts_seq_len, insts):
            for i in range(seq_len):
                insts_5gram.append([cls_id]+inst[i:i+5]+[sep_id])
        insts = torch.tensor(insts_5gram).to(self.device)
        assert insts.shape[0] == sum(insts_seq_len) and insts.shape[1] == 7, 'insts tokenizing goes wrong.'
        return insts, insts_seq_len

    # def __tokenize(self, batch_size: int, seq_len: int, insts: List[str], golds: torch.Tensor):
    #     insts = [self.tokenizer.encode(inst) + [self.tokenizer.pad_token_id] * (seq_len-(len(inst)+1)//2) for inst in insts]
    #     insts = torch.tensor(insts).to(self.device)
    #     assert insts.shape[0] == batch_size and insts.shape[1] == seq_len + 2, 'insts tokenizing goes wrong.'
    #     mask = torch.ones((batch_size, seq_len + 2), dtype=torch.long).to(self.device)
    #     mask[:, 2:] = golds != Constants.actionPadId
    #     return insts, mask

    def __init_para(self):
        init.xavier_uniform_(self.mlp[0].weight)
        init.uniform_(self.mlp[0].bias)
        init.xavier_uniform_(self.cls.weight)
        init.uniform_(self.cls.bias)
        # init.xavier_uniform_(self.transformer.linear1.weight)
        # init.xavier_normal_(self.transformer.linear1.bias)
        # init.xavier_uniform_(self.transformer.linear2.weight)
        # init.xavier_normal_(self.transformer.linear2.bias)

    def pack_state_dict(self):
        res = {'dropo': self.dropout1.state_dict(),
               # 'transf': self.transformer.state_dict(),
               'cnn': self.cnn.state_dict(),
               'mlp': self.mlp.state_dict(), 'cls': self.cls.state_dict()}
        return res

