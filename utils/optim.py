# @Author : guopeiming
# @Datetime : 2019/10/18 08:33
# @File : train.py
# @Last Modify Time : 2019/11/06 19:03
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch.optim


class Optim:
    """
    My Optim class
    """
    name_list = ['Adam']

    def __init__(self, name, learning_rate, weight_decay, model):
        assert name in Optim.name_list, 'optimizer name is wrong.'
        if name == 'Adam':
            self._optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()

