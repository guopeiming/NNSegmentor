# @Author : guopeiming
# @Datetime : 2019/10/19 15:33
# @File : train.py
# @Last Modify Time : 2019/10/19 15:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import os
from torch.utils.tensorboard import SummaryWriter
from config.Constants import embeddings_layer_keyword


class VisualLogger:
    def __init__(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.writer = SummaryWriter(path)

    def visual_scalars(self, dic, step, typ):
        for tag in dic:
            self.writer.add_scalar(tag+'/'+typ, dic[tag], step)

    def visual_histogram(self, model, step):
        for tag, values in model.named_parameters():
            tag = tag.replace('.', '/')
            # if tag.find(embeddings_layer_keyword) == -1:
            #     self.writer.add_histogram(tag, values.data, step)
            # else:
            #     self.writer.add_embedding(values.data, global_step=step, tag=tag)
            self.writer.add_histogram(tag, values.data, step)
            if values.data.grad is not None:
                self.writer.add_histogram(tag+'grad', values.data.grad, step)

    def visual_graph(self, model, input_to_model, verbose=False):
        self.writer.add_graph(model, input_to_model, verbose)

    def close(self):
        self.writer.close()

