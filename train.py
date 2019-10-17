# @Author : guopeiming
# @Datetime : 2019/10/12 18:18
# @File : train.py
# @Last Modify Time : 2019/10/16 13:46
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import argparse
from config.config import MyConf
from utils.pretrained_embed import load_pretrained_embed
from torch.utils.data import DataLoader
from utils.dataset import CWSDataset, pad_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="NNTranSegmentor")
    parser.add_argument("--config", dest="config", type=str, required=False, default="./config/config.cfg",
                        help="path of config_file")
    args = parser.parse_args()
    config = MyConf(args.config)
    return config


def load_data(config):
    if not config.fine_tune:
        config.pretrained_embedding = load_pretrained_embed(config)
    data = torch.load(config.data_path)
    train_data = data["data"]["train"]
    train_data = DataLoader(dataset=CWSDataset(data["dic"], train_data, config), batch_size=config.batch_size,
                            num_workers=config.num_workers, collate_fn=pad_collate_fn, drop_last=config.drop_last)
    dev_data = data["data"]["dev"]
    dev_data = DataLoader(dataset=CWSDataset(data["dic"], dev_data, config), batch_size=config.batch_size,
                          shuffle=config.shuffle, num_workers=config.num_workers, collate_fn=pad_collate_fn,
                          drop_last=config.drop_last)
    test_data = data["data"]["test"]
    test_data = DataLoader(dataset=CWSDataset(data["dic"], test_data, config), batch_size=config.batch_size,
                           shuffle=config.shuffle, num_workers=config.num_workers, collate_fn=pad_collate_fn,
                           drop_last=config.drop_last)
    print('train_dataset, dev_dataset, test_dataset loading completes.\n')
    return train_data, dev_data, test_data


def main():
    config = parse_args()

    # ========= Loading Dataset ========= #
    print("Loading dataset starts...")
    # load_data(config)
    # train_data, dev_data, test_data = load_data(config)

    # ========= Preparing Model ========= #
    print("Preparing Model starts...")
    print(config)
    config.device = torch.device('cpu')
    if config.use_cuda:
        assert torch.cuda.is_available(), "Cuda is not available."
        config.device = torch.device('cuda:0')
    # print(next(iter(train_data)))


if __name__ == '__main__':
    main()

