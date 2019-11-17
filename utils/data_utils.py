# @Author : guopeiming
# @Datetime : 2019/10/15 16:11
# @File : train.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from config import Constants
from torch.utils.data import DataLoader
from utils.dataset import CWSDataset, pad_collate_fn


def gen_embed_nnembed(embeddings_dic, id2item, length):
    oov_num = 0
    embed = nn.Embedding(len(id2item), length, padding_idx=Constants.padId).weight
    init.xavier_uniform_(embed)
    embed = embed.numpy()
    for idx, item in enumerate(id2item):
        if item in embeddings_dic:
            embed[idx] = np.array(embeddings_dic[item], dtype=np.float32)
        else:
            oov_num += 1
    assert oov_num < Constants.TEST_OOV_NUM, 'The number of oov is too big.'
    print('The number of (oov) is %d' % oov_num)
    return torch.tensor(embed)


def read_embed_file(id2item, filename, length, gen_oov_mode, uniform_par):
    assert gen_oov_mode in ['zeros', 'nnembed', 'avg', 'uniform'], \
        'Mode of generating oov embedding vector does not exist.'
    embeddings_dic = {}
    with open(filename, mode='r', encoding='utf-8') as reader:
        lines = reader.readlines()
        assert length == len(lines[0].strip().split(' ')) - 1, 'Pretrained embeddings dimension is correct.'
        for line in lines:
            strs = line.strip().split(' ')
            if strs[0] in id2item:
                embeddings_dic[strs[0]] = [float(value) for value in strs[1:]]
    if Constants.padKey not in embeddings_dic:
        embeddings_dic[Constants.padKey] = [0.] * length
        print('pretrained embeddings does not include <pad>, init it zeros default.')
    if gen_oov_mode == 'nnembed':
        return gen_embed_nnembed(embeddings_dic, id2item, length)
    else:
        avg = np.mean(np.array([embeddings_dic[key] for key in embeddings_dic], dtype=np.float32), axis=0)
        embed = np.empty((len(id2item), length), dtype=np.float32)
        oov_num = 0
        for idx, item in enumerate(id2item):
            if item in embeddings_dic:
                embed[idx] = np.array(embeddings_dic[item], dtype=np.float32)
            else:
                if gen_oov_mode == 'zeros':
                    embed[idx] = np.array([0.] * length, dtype=np.float32)
                elif gen_oov_mode == 'uniform':
                    embed[idx] = np.random.uniform(-uniform_par, uniform_par, size=length)
                else:
                    embed[idx] = avg
                oov_num += 1
                print('pretrained embeddings does not include %s, init it by %s default.' % (item, gen_oov_mode))
        assert oov_num < Constants.MAX_OOV_NUM, 'The number of oov is too big.'
        print('The number of (oov) is %d' % oov_num)
        return torch.tensor(embed)


def load_pretrained_embeddings(train_dataset, config):
    print('Loading char embeddings starts...')
    if config.pretrained_embed_char:
        print('Loading char pretrained embeddings from %s' % config.pretrained_char_embed_file)
        char_embeddings = read_embed_file(train_dataset.get_id2char(), config.pretrained_char_embed_file,
                                          config.char_embed_dim, config.char_gen_oov_mode, config.char_gen_oov_uniform)
    else:
        char_embeddings = init.xavier_uniform_(nn.Embedding(train_dataset.get_char_vocab_size(), config.char_embed_dim,
                                                            padding_idx=Constants.padId, max_norm=config.char_embed_max_norm).weight)
        print('char pretrained embeddings was loaded by random.')
    print('Loading char embeddings ends.\n')

    print('Loading bichar embeddings starts...')
    if config.pretrained_embed_bichar:
        print('Loading bichar pretrained embeddings from %s' % config.pretrained_bichar_embed_file)
        bichar_embeddings = read_embed_file(train_dataset.get_id2bichar(), config.pretrained_bichar_embed_file,
                                            config.bichar_embed_dim, config.bichar_gen_oov_mode, config.bichar_gen_oov_uniform)
    else:
        bichar_embeddings = init.xavier_uniform_(nn.Embedding(train_dataset.get_bichar_vocab_size(), config.bichar_embed_dim,
                                                            padding_idx=Constants.padId, max_norm=config.bichar_embed_max_norm).weight)
    print('Loading bichar embeddings ends.\n')
    return char_embeddings, bichar_embeddings


def load_data(config):
    data = torch.load(config.data_path)
    train_data = data["data"]["train"]
    train_dataset = CWSDataset(data["dic"], train_data)
    train_data = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                            num_workers=config.num_workers, collate_fn=pad_collate_fn, drop_last=config.drop_last)
    dev_data = data["data"]["dev"]
    dev_data = DataLoader(dataset=CWSDataset(data["dic"], dev_data), batch_size=config.batch_size,
                          shuffle=config.shuffle, num_workers=config.num_workers, collate_fn=pad_collate_fn,
                          drop_last=config.drop_last)
    test_data = data["data"]["test"]
    test_data = DataLoader(dataset=CWSDataset(data["dic"], test_data), batch_size=config.batch_size,
                           shuffle=config.shuffle, num_workers=config.num_workers, collate_fn=pad_collate_fn,
                           drop_last=config.drop_last)
    print('train_dataset, dev_dataset, test_dataset loading completes.')
    return train_data, dev_data, test_data, train_dataset

