# @Author : guopeiming
# @Contact : guopeiming2016@{qq.com, gmail.com, 163.com}
from typing import List
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class MyLRScheduler(LambdaLR):
    """
    My lr scheduler class, it can convert to warm_up, no_warm_up, constant, warm_up_reduce learning rate mode
    by parameters.
    """
    def __init__(self, optimizer: Optimizer, lr_lambdas: List, last_epoch=-1):
        self.curr_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
        super(MyLRScheduler, self).__init__(optimizer, lr_lambdas, last_epoch)

    def get_lr(self):
        self.curr_lrs = [lmbda(self.last_epoch, lr) for lmbda, lr in zip(self.lr_lambdas, self.curr_lrs)]
        return self.curr_lrs.copy()


def get_lr_scheduler_lambda(init_lr, warmup_step, decay_factor):
    return lambda step, curr_lr: (step/warmup_step)*init_lr if step <= warmup_step else curr_lr**decay_factor

