# @Author : guopeiming
# @Datetime : 2019/10/18 08:33
# @File : optim.py
# @Last Modify Time : 2019/11/28 19:03
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch.optim
from collections.abc import Iterable
import torch.nn.utils as utils
from utils.MyLRScheduler import MyLRScheduler, get_lr_scheduler_lambda


class Optim:
    """
    My Optim class
    """
    name_list = ['Adam', 'SGD']

    def __init__(self, name, learning_rate, weight_decay, model, config):
        assert name in Optim.name_list, 'optimizer name is wrong.'
        if name == 'Adam':
            self._optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if name == 'SGD':
            self._optimizer = torch.optim.SGD(model.parameters(), learning_rate, config.momentum, config.dampening,
                                              weight_decay, config.nesterov)

        self.model = model
        self.clip_grad = config.clip_grad
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.scheduler = MyLRScheduler(self._optimizer, get_lr_scheduler_lambda(learning_rate, config.warmup_steps, config.lr_decay_factor))

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        if self.clip_grad:
            utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_max_norm)

        self._optimizer.step()

        self.scheduler.step()

    def get_optimizer(self):
        return self._optimizer

    def get_lr(self):
        return self.scheduler.get_lr()

    def set_freeze_by_idxs(self, idxs, free):
        if not isinstance(idxs, Iterable):
            idxs = [idxs]

        for name, model_layer in self.model.bert.encoder.layer.named_children():
            if name not in idxs:
                continue
            for param in model_layer.parameters():
                param.requires_grad_(not free)

