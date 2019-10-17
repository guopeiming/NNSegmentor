# @Author : guopeiming
# @Datetime : 2019/10/15 16:11
# @File : train.py
# @Last Modify Time : 2019/10/17 09:27
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from config import Constants


def gen_embed_nnembed(embeddings_dic, id2item, length):
    oov_num = 0
    embed = nn.Embedding(len(id2item), length, padding_idx=Constants.padId).weight.data
    init.xavier_uniform(embed)
    embed = embed.numpy()
    for idx, item in enumerate(id2item):
        if item in embeddings_dic:
            embed[idx] = np.array(embeddings_dic[item], dtype=float)
        else:
            oov_num += 1
    assert oov_num < Constants.TEST_OOV_NUM, 'The number of oov is too big.'
    print('The number of (oov) is %d' % oov_num)
    return torch.tensor(embed)


def read_embed_file(id2item, filename, length, gen_oov_mode, uniform_par):
    assert gen_oov_mode in ['zeros', 'nnembed', 'avg', 'uniform'], \
        'Mode of generating oov embedding vector does not exist.'
    embeddings_dic = {}
    with open(filename, mode='r') as reader:
        lines = reader.readlines()
        assert length == len(lines[0].strip().strip(' ')) - 1, 'Pretrained embeddings dimension is correct.'
        for line in lines:
            strs = line.strip().split(' ')
            if strs[0] in id2item:
                embeddings_dic[strs[0]] = [float(value) for value in strs[1:]]
    if Constants.padKey not in embeddings_dic:
        embeddings_dic[Constants.padKey] = [0.] * length
    if gen_oov_mode == 'nnembed':
        return gen_embed_nnembed(embeddings_dic, id2item, length)
    else:
        avg = np.mean(np.array([embeddings_dic[key] for key in embeddings_dic], dtype=float), axis=0)
        embed = np.empty((len(id2item), length), dtype=float)
        oov_num = 0
        for idx, item in enumerate(id2item):
            if item in embeddings_dic:
                embed[idx] = np.array(embeddings_dic[item], dtype=float)
            else:
                if gen_oov_mode == 'zeros':
                    embed[idx] = np.array([0.] * length, dtype=float)
                elif gen_oov_mode == 'uniform':
                    embed[idx] = np.random.uniform(-uniform_par, uniform_par, size=length)
                else:
                    embed[idx] = avg
                oov_num += 1
        assert oov_num < Constants.TEST_OOV_NUM, 'The number of oov is too big.'
        print('The number of (oov) is %d' % oov_num)
        return torch.tensor(embed)


def load_pretrained_char_embed(dataset, config):
    print('Loading char embeddings starts...')
    print('Loading %s pretrained embeddings from %s' % ('char', config.char_embed_file))
    embeddings = read_embed_file(dataset.get_id2char, config.char_embed_file, config.char_embed_dim,
                                 config.char_gen_oov_mode, config.char_gen_oov_uniform)
    print('Loading char embeddings ends.')
    return embeddings


def load_pretrained_word_embed(dataset, config):
    print('Loading word embeddings starts...')
    print('Loading %s pretrained embeddings from %s' % ('word', config.word_embed_file))
    embeddings = read_embed_file(dataset.get_id2word, config.word_embed_file, config.word_embed_dim,
                                 config.word_gen_oov_mode, config.word_gen_oov_uniform)
    print('Loading word embeddings ends.')
    return embeddings

