# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch.optim
import torch.nn.utils as utils
from config.config import MyConf
from collections.abc import Iterable
from config.Constants import bertAttr
from model.BertCharSegmentor import BertCharSegmentor
from utils.MyLRScheduler import MyLRScheduler, get_lr_scheduler_lambda


class Optim:
    """
    My Optim class
    """
    name_list = ['Adam', 'SGD']

    def __init__(self, model: BertCharSegmentor, config: MyConf):
        assert config.opti_name in Optim.name_list, 'optimizer name is wrong.'

        params_list = []
        lr_lambdas = []
        for name, child in model.named_children():
            if bertAttr in name:
                params_list.append({'params': child.parameters(), 'lr': config.fine_tune_lr})
                lr_lambdas.append(get_lr_scheduler_lambda(config.fine_tune_lr, config.warmup_steps, config.lr_decay_factor))
            else:
                params_list.append({'params': child.parameters(), 'lr': config.learning_rate})
                lr_lambdas.append(get_lr_scheduler_lambda(config.learning_rate, config.warmup_steps, config.lr_decay_factor))

        if config.opti_name == 'Adam':
            self._optimizer = torch.optim.Adam(params_list, weight_decay=config.weight_decay)
        if config.opti_name == 'SGD':
            self._optimizer = torch.optim.SGD(params_list, config.learning_rate, config.momentum, config.dampening,
                                              config.weight_decay, config.nesterov)

        self.model = model
        self.clip_grad = config.clip_grad
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.scheduler = MyLRScheduler(self._optimizer, lr_lambdas)

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

        for name, model_layer in self.model.bert_model.encoder.layer.named_children():
            if name not in idxs:
                continue
            for param in model_layer.parameters():
                param.requires_grad_(not free)

    def freeze_pooler(self, free=True):
        for param in self.model.bert_model.pooler.parameters():
            param.requires_grad_(not free)

    def free_embeddings(self, free=True):
        for param in self.model.bert_model.embeddings.parameters():
            param.requires_grad_(not free)

