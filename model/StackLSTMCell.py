# @Author : guopeiming
# @Datetime : 2019/10/31 22:13
# @File : dataset.py
# @Last Modify Time : 2019/10/31 22:13
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import numpy as np
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
        self.idx = None
        self.pos = None
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size, bias=True)
        self.__init_para()

    def forward(self, insts, op):
        """

        :param insts: (batch_size, self.input_size) input subword embeddings tensor, in batch
        :param op: (batch_size, ) input op tensor, -1 means pop, 1 means push, 0 means hold
        :return:
        """
        # hx, cx = self.stack_hidden.gather(1, pos).squeeze(1), self.stack_cell.gather(1, pos).squeeze(1)
        # self.stack_hidden[]
        return insts

    def build_stack(self, stack_size, batch_size):
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
        init.normal_(self.lstm.weight_hh, mean=0., std=np.sqrt(6./(self.input_size+self.hidden_size)))
        init.normal_(self.lstm.weight_ih, mean=0., std=np.sqrt(6./(self.input_size+self.hidden_size)))
        init.normal_(self.lstm.bias_hh, mean=0., std=np.sqrt(6./(self.input_size+self.hidden_size)))
        init.normal_(self.lstm.bias_ih, mean=0., std=np.sqrt(6./self.input_size+self.hidden_size))

