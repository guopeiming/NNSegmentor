# @Author : guopeiming
# @Datetime : 2019/10/31 22:13
# @File : dataset.py
# @Last Modify Time : 2019/10/31 22:13
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import torch.nn as nn
import torch.nn.init as init


class WordStackLSTMCell(nn.Module):
    """
    StackLSTMCell
    """
    def __init__(self, input_size, hidden_size, device):
        super(WordStackLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.stack_hidden = None
        self.stack_cell = None
        self.idx = None
        self.pos = None
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size, bias=True)
        self.__init_para()

    def forward(self, subword):
        """
        :param subword: (batch_size, self.input_size) input subword embeddings tensor, in batch
        :return: output of word_decoder layer
        """
        h, c = self.stack_hidden[self.idx, self.pos, :], self.stack_cell[self.idx, self.pos, :]
        h, c = self.lstm(subword, (h, c))
        self.stack_hidden[self.idx, self.pos + 1, :], self.stack_cell[self.idx, self.pos + 1, :] = h, c
        return h, c

    def update_pos(self, op):
        """
        update self.pos
        :param op: (batch_size, ) op vector, -1 means pop, 0 means hold, 1 means push.
        :return:
        """
        self.pos = self.pos + op

    def init_stack(self, stack_size, batch_size):
        """
        build stack when batch training starts.

        :param stack_size: int, the stack size.
        :param batch_size: int, the batch size.
        :return:
        """
        self.stack_hidden = torch.zeros((batch_size, stack_size, self.hidden_size)).to(self.device)
        self.stack_cell = torch.zeros((batch_size, stack_size, self.hidden_size)).to(self.device)
        self.idx = torch.arange(batch_size, dtype=torch.long).to(self.device)
        self.pos = torch.zeros(batch_size, dtype=torch.long).to(self.device)

    def __init_para(self):
        init.xavier_normal_(self.lstm.weight_hh)
        init.xavier_normal_(self.lstm.weight_ih)
        init.normal_(self.lstm.bias_hh)
        init.normal_(self.lstm.bias_ih)

