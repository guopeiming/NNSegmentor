# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import torch
from config import Constants
from config.config import MyConf
from typing import List
from torch.utils.data import Dataset, DataLoader


class CWSBertDataset(Dataset):
    def __init__(self, insts, golds, dict_datas: List[List[List[int]]]):
        super(CWSBertDataset, self).__init__()
        self.insts = insts
        self.golds = golds
        self.dict_datas = dict_datas

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, idx):
        return [self.insts[idx], self.golds[idx], self.dict_datas[idx]]


def pad_collate_fn(insts):
    """
    Pad the instance to the max seq length in batch
    """
    insts, golds, dict_datas = list(zip(*insts))
    max_len = max(len(gold) for gold in golds)

    golds = torch.tensor([gold + [Constants.actionPadId] * (max_len - len(gold)) for gold in golds], dtype=torch.long)
    dict_datas = torch.tensor([dict_data+[[1]*15]*(max_len-len(dict_data)) for dict_data in dict_datas], dtype=torch.float)
    return [insts, golds, dict_datas]


def load_data(config: MyConf):
    data = torch.load(config.data_path)
    dict_data = torch.load(config.dict_data_path)
    train_dataset = CWSBertDataset(data['train_insts'], data['train_golds'], dict_data['train'])
    dev_dataset = CWSBertDataset(data['dev_insts'], data['dev_golds'], dict_data['dev'])
    test_dataset = CWSBertDataset(data['test_insts'], data['test_golds'], dict_data['test'])
    train_data = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                            num_workers=config.num_workers, collate_fn=pad_collate_fn, drop_last=config.drop_last)
    dev_data = DataLoader(dataset=dev_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                          num_workers=config.num_workers, collate_fn=pad_collate_fn, drop_last=config.drop_last)
    test_data = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                           num_workers=config.num_workers, collate_fn=pad_collate_fn, drop_last=config.drop_last)
    print('train_dataset, dev_dataset, test_dataset loading completes.')
    return train_data, dev_data, test_data, train_dataset

