# @Author : guopeiming
# @Datetime : 2019/10/12 18:59
# @File : dataset.py
# @Last Modify Time : 2019/11/16 14:25
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
from config import Constants
from torch.utils.data.dataset import Dataset


class CWSDataset(Dataset):
    """
    dataset for training NNTranSegmentor
    """
    def __init__(self, dic, data):
        super(CWSDataset, self).__init__()
        assert len(data['insts']) == len(data['golds']), "The number of insts and golds must be equal."
        self.char2id = dic['char2id']
        self.id2char = dic['id2char']
        self.bichar2id = dic['bichar2id']
        self.id2bichar = dic['id2bichar']
        self.insts_char = data['insts_char']
        self.insts_bichar_l = data['insts_bichar_l']
        self.insts_bichar_r = data['insts_bichar_r']
        self.golds = data['golds']

    def __len__(self):
        return len(self.insts_char)

    def __getitem__(self, idx):
        inst_char = self.insts_char[idx]
        inst_bichar_l = self.insts_bichar_l[idx]
        inst_bichar_r = self.insts_bichar_r[idx]
        gold = self.golds[idx]
        return [inst_char, inst_bichar_l, inst_bichar_r, gold]

    def get_char_vocab_size(self):
        return len(self.id2char)

    def get_bichar_vocab_size(self):
        return len(self.id2bichar)

    def get_id2char(self):
        return self.id2char

    def get_id2bichar(self):
        return self.id2bichar

    def get_char2id(self):
        return self.char2id

    def get_bichar2id(self):
        return self.bichar2id


def pad_collate_fn(insts):
    """
    Pad the instance to the max seq length in batch
    """
    insts_char, insts_bichar_l, insts_bichar_r, golds = list(zip(*insts))
    max_len = max(len(inst_char) for inst_char in insts_char)

    insts_char = torch.tensor([inst_char + [Constants.padId] * (max_len - len(inst_char)) for inst_char in insts_char])
    insts_bichar_l = torch.tensor([inst_bichar_l + [Constants.padId] * (max_len - len(inst_bichar_l)) for inst_bichar_l in insts_bichar_l])
    insts_bichar_r = torch.tensor([inst_bichar_r + [Constants.padId] * (max_len - len(inst_bichar_r)) for inst_bichar_r in insts_bichar_r])
    golds = torch.tensor([gold + [Constants.actionPadId] * (max_len - len(gold)) for gold in golds], dtype=torch.long)
    return [(insts_char, insts_bichar_l, insts_bichar_r), golds]

