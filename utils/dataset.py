# @Author : guopeiming
# @Datetime : 2019/10/12 18:59
# @File : dataset.py
# @Last Modify Time : 2019/10/18 08:33
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
        self.word2id = dic['word2id']
        self.id2word = dic['id2word']
        self.insts = data['insts']
        self.golds = data['golds']

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        inst = self.insts[idx]
        gold = self.golds[idx]
        # if not self.fine_tune:
        #     dic = self.pretrained_embed['char']['item2id']
        #     inst = [dic[self.id2char[id_]] if self.id2char[id_] in dic else Constants.oovId for id_ in inst]
        return [inst, gold]

    def get_char_vocab_size(self):
        return len(self.id2char)

    def get_word_vocab_size(self):
        return len(self.id2word)

    def get_id2char(self):
        return self.id2char

    def get_id2word(self):
        return self.id2word

    def get_char2id(self):
        return self.char2id

    def get_word2id(self):
        return self.word2id


def pad_collate_fn(insts):
    """
    Pad the instance to the max seq length in batch
    """
    insts, golds = list(zip(*insts))
    max_len = max(len(inst) for inst in insts)

    insts = torch.tensor([inst + [Constants.padId] * (max_len - len(inst)) for inst in insts], dtype=torch.int64)
    return [insts, golds]

