# @Author : guopeiming
# @Datetime : 2019/10/19 15:33
# @File : train.py
# @Last Modify Time : 2019/10/19 15:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import os
from torch.utils.tensorboard import SummaryWriter


class VisualLogger:
    def __init__(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.writer = SummaryWriter(path)

    def visual_scalars(self, dic, step):
        for tag in dic:
            self.writer.add_scalar(tag, dic[tag], step)

    def close(self):
        self.writer.close()

