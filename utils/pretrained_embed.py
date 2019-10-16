# @Author : guopeiming
# @Datetime : 2019/10/15 16:11
# @File : train.py
# @Last Modify Time : 2019/10/16 14:01
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import numpy as np
from config import Constants


def read_embed_file(filename):
    item2id = {}
    id2item = []
    with open(filename) as reader:
        lines = reader.readlines()
        embed = np.zeros(len(lines), len(lines[0].split(' ')) - 1)
        for idx, line in enumerate(lines):
            line = line.split(' ')
            item2id[line[0]] = idx
            id2item[idx] = line[0]
            embed[idx] = np.array([float(value) for value in line[1:]], dtype='float64')
    assert id2item[Constants.oovId] == Constants.oovKey, 'oovId in pretrained embeddings is wrong.'
    assert id2item[Constants.padId] == Constants.padKey, 'padId in pretrained embeddings is wrong.'
    return {'item2id': item2id, 'id2item': id2item, 'embed': embed}


def load_pretrained_embed(config):
    print('Loading embeddings starts...')
    dic = {}
    print('Loading %s pretrained embedding from %s' % ('char', config.char_embed_file))
    dic['char'] = read_embed_file(config.char_embed_file)
    print('Loading %s pretrained embedding from %s' % ('word', config.word_embed_file))
    dic['word'] = read_embed_file(config.word_embed_file)
    return dic

