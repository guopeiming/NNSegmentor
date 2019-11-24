# @Author : guopeiming
# @Datetime : 2019/10/15 16:11
# @File : train.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import torch.nn as nn
import torch.nn.init as init
from config import Constants
from torch.utils.data import DataLoader
from utils.dataset import CWSDataset, pad_collate_fn


def build_embed_tensor_from_file(embeddings, filename, vocab):
    embed_size = embeddings.shape[1]
    hit_set = set()
    with open(filename, mode='r', encoding='utf-8') as file:
        for line in file:
            cont_list = line.strip().split(' ')
            if cont_list[0] in vocab:
                assert embed_size == len(cont_list) - 1, 'pretrained embeddings size error.'
                embeddings[vocab[cont_list[0]]] = torch.tensor([float(x) for x in cont_list[1:]])
                hit_set.add(cont_list[0])
    return embeddings, hit_set


def get_pretr_embed_tensor(flag, filename, vocab, embed_dim, gen_oov_mode, gen_oov_uniform, max_norm, typ):
    print('Loading %s embeddings starts...' % typ)
    if flag:
        assert gen_oov_mode in ['zeros', 'nnembed', 'avg', 'uniform', 'randn'], \
            'Mode of generating oov embedding vector does not exist.'
        if gen_oov_mode == 'nnembed':
            embeddings = torch.nn.Embedding(len(vocab), embed_dim, Constants.padId, max_norm).weight
        elif gen_oov_mode == 'uniform':
            embeddings = init.uniform_(torch.randn((len(vocab), embed_dim)), -gen_oov_uniform, gen_oov_uniform)
        elif gen_oov_mode == 'randn':
            embeddings = torch.randn((len(vocab), embed_dim))
        else:
            embeddings = torch.zeros((len(vocab), embed_dim))
        print('Loading %s pretrained embeddings from %s' % (typ, filename))
        embeddings, hit_set = build_embed_tensor_from_file(embeddings, filename, vocab)
        print('%d %s are hit in pretrained embeddings file' % (len(hit_set), typ))
        print('%d %s are OOV, which are inited by %s' % ((len(vocab) - len(hit_set)), typ, gen_oov_mode))
        print('OOV %s:' % typ)
        oov_set = set(vocab.keys()).difference(hit_set)
        print(oov_set)
        assert (len(vocab) - len(hit_set)) <= Constants.MAX_OOV_NUM, 'OOV num too big.'
        if gen_oov_mode == 'avg':
            avg_tensor = torch.mean(embeddings, 0)
            for key in oov_set:
                embeddings[vocab[key]] = avg_tensor.clone().detach()
    else:
        embeddings = init.xavier_uniform_(nn.Embedding(len(vocab), embed_dim, Constants.padId, max_norm).weight)
        print('%s pretrained embeddings was loaded by random.' % typ)
    print('Loading %s embeddings ends.\n' % typ)
    return embeddings


def load_pretrained_embeddings(train_dataset, config):
    char_embeddings = get_pretr_embed_tensor(config.pretrained_embed_char, config.pretrained_char_embed_file,
                                             train_dataset.get_char2id(), config.char_embed_dim,
                                             config.char_gen_oov_mode, config.char_gen_oov_uniform,
                                             config.char_embed_max_norm, 'char')
    bichar_embeddings = get_pretr_embed_tensor(config.pretrained_embed_bichar, config.pretrained_bichar_embed_file,
                                               train_dataset.get_bichar2id(), config.bichar_embed_dim,
                                               config.bichar_gen_oov_mode, config.bichar_gen_oov_uniform,
                                               config.bichar_embed_max_norm, 'bichar')
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


def get_lr_scheduler_lambda(warmup_step, decay_factor):
    return lambda step: step/warmup_step if step <= warmup_step else decay_factor**(step-warmup_step)

