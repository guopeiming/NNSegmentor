# @Author : guopeiming
# @Datetime : 2019/10/31 22:13
# @File : dataset.py
# @Last Modify Time : 2019/10/31 22:13
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import torch.nn as nn
import torch.nn.init as init


class StackLSTMCell(nn.Module):
    """
    StackLSTMCell
    """
    def __init__(self, input_size, hidden_size, device):
        super(StackLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.stack_hidden = None
        self.stack_cell = None
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size, bias=True)
        self.__init_para()

    def forward(self, input):
        """
        :param input: (batch_size, self.input_size) input tensor, in batch
        :return:
        """
        h, c = self.lstm(input, (torch.cat(self.stack_hidden, 0), torch.cat(self.stack_cell, 0)))
        return h, c

    def update_pos(self, op, h, c):
        """
        update self.pos
        :param op: (batch_size, ) op vector, 0 means hold, 1 means move, 2 means set zeros.
        :return:
        """
        for i in range(op.shape[0]):
            if op[i] == 1:
                self.stack_hidden[i] = h[i].unsqueeze(0)
                self.stack_cell[i] = c[i].unsqueeze(0)
            if op[i] == 2:
                self.stack_hidden[i] = torch.zeros((1, self.hidden_size)).to(self.device)

    def init_stack(self, batch_size):
        """
        build stack when batch training starts.

        :param stack_size: int, the stack size.
        :param batch_size: int, the batch size.
        :return:
        """
        self.stack_hidden = list(torch.zeros((batch_size, self.hidden_size)).to(self.device).chunk(batch_size, 0))
        self.stack_cell = list(torch.zeros((batch_size, self.hidden_size)).to(self.device).chunk(batch_size, 0))
        # self.idx = torch.arange(batch_size, dtype=torch.long).to(self.device)
        # self.pos = torch.zeros(batch_size, dtype=torch.long).to(self.device)

    def __init_para(self):
        init.xavier_normal_(self.lstm.weight_hh)
        init.xavier_normal_(self.lstm.weight_ih)
        init.normal_(self.lstm.bias_hh)
        init.normal_(self.lstm.bias_ih)

