# @Author : guopeiming
# @Datetime : 2019/10/12 18:59
# @File : dataset.py
# @Last Modify Time : 2019/10/15 18:59
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import numpy as np
from config import Constants
from torch.utils.data.dataset import Dataset


class CWSDataset(Dataset):
    """
    dataset for training NNTranSegmentor
    """
    def __init__(self, dic, data, config):
        super(CWSDataset, self).__init__()
        assert len(data['insts']) == len(data['golds']), "The number of insts and golds must be equal."
        self.char2id = dic['char2id']
        self.id2char = dic['id2char']
        self.word2id = dic['word2id']
        self.id2word = dic['id2word']
        self.insts = data['insts']
        self.golds = data['golds']
        self.fine_tune = config.fine_tune
        self.pretrained_embed = None

        if not self.fine_tune:
            self.pretrained_embed = config.pretrained_embedding

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        inst = self.insts[idx]
        gold = self.golds[idx]
        if not self.fine_tune:
            dic = self.pretrained_embed['char']['item2id']
            inst = [dic[self.id2char[id_]] if self.id2char[id_] in dic else Constants.oovId for id_ in inst]
        return [inst, gold]


def pad_collate_fn(insts):
    """
    Pad the instance to the max seq length in batch
    """
    insts, golds = list(zip(*insts))
    max_len = max(len(inst) for inst in insts)

    insts = torch.tensor([inst + [Constants.padId] * (max_len - len(inst)) for inst in insts], dtype=torch.int64)
    return [insts, golds]

