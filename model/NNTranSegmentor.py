# @Author : guopeiming
# @Datetime : 2019/10/16 14:17
# @File : dataset.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}


import torch.nn as nn
from model.char_encoder import CharEncoder
from model.word_decoder import WordDecoder


class NNTranSegmentor(nn.Module):
    """
    NNTranSegmentor
    """

    def __init__(self, id2char, word2id, char_vocab_size, word_vocab_size, config):
        super(NNTranSegmentor, self).__init__()

        self.char_encoder = CharEncoder(char_vocab_size, id2char, config)
        self.word_decoder = WordDecoder(word_vocab_size, id2char, word2id, config)

    def forward(self, insts, golds=None):
        char_context = self.char_encoder(insts)  # [char_num, batch, embeddings_dim]
        pred = self.word_decoder(char_context, insts, golds)  # [batch, char_num, label_num]
        return pred
