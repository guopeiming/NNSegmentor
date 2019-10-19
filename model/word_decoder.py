# @Author : guopeiming
# @Datetime : 2019/10/16 14:17
# @File : dataset.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from config import Constants
import torch.nn.functional as F
from utils.data_utils import load_pretrained_word_embed


class WordDecoder(nn.Module):
    """
    submodel of NNTranSegmentor ------ word_decoder
    """

    def __init__(self, word_vocab_size, id2char, word2id, config):
        super(WordDecoder, self).__init__()

        # word embed layer
        if config.pretrained_embed_word:
            embed = load_pretrained_word_embed(word2id, config)
            self.word_embed = nn.Embedding.from_pretrained(embed, freeze=config.freeze, padding_idx=Constants.padId,
                                                           max_norm=config.word_embed_max_norm)
        else:
            self.word_embed = nn.Embedding(word_vocab_size, config.word_embed_dim,
                                           padding_idx=Constants.padId,
                                           max_norm=config.word_embed_max_norm)
            self.word_embed.weight.requires_grad_(config.fine_tune)
            init.normal_(self.word_embed.weight, mean=0.0, std=np.sqrt(5. / config.char_embed_dim))

        # word lstm layer
        self.lstm = nn.LSTMCell(config.word_embed_dim, config.word_lstm_hid_dim, True)
        init.normal_(self.lstm.weight_ih, mean=0.0, std=np.sqrt(5./config.word_embed_dim))
        init.normal_(self.lstm.weight_hh, mean=0.0, std=np.sqrt(5./config.word_embed_dim))
        init.uniform_(self.lstm.bias_hh)
        init.uniform_(self.lstm.bias_ih)

        # compose layer map some char to subword
        self.char_compose = nn.Linear(config.char_lstm_hid_dim*2, config.word_embed_dim, bias=True)
        init.xavier_normal_(self.char_compose.weight)
        init.uniform_(self.char_compose.bias)

        # linear layer compose char and word feature to label
        self.compose = nn.Linear(config.word_lstm_hid_dim+config.char_lstm_hid_dim*2, 3, bias=True)
        init.xavier_normal_(self.char_compose.weight)
        init.uniform_(self.char_compose.bias)

        # char_embed_dim, word_embed_dim
        self.char_embed_dim = config.char_embed_dim
        self.word_embed_dim = config.word_embed_dim
        self.char_lstm_hid_dim = config.char_lstm_hid_dim
        self.word_lstm_hid_dim = config.word_lstm_hid_dim
        self.id2char = id2char
        self.word2id = word2id
        # self.use_cuda = config.use_cuda
        # self.device = config.device

    def forward(self, char_context, insts, golds):
        char_idx_list = [0]*char_context.shape[1]
        h = [torch.zeros((1, self.word_lstm_hid_dim), dtype=torch.float)] * char_context.shape[1]
        c = [torch.zeros((1, self.word_lstm_hid_dim), dtype=torch.float)] * char_context.shape[1]
        # if self.use_cuda:
        #     h.to(self.device)
        #     c.to(self.device)

        pred_list = []
        for idx in range(char_context.shape[0]):
            subwords = self._compose_subword(idx, char_idx_list, char_context)
            words_out, c_ = self.lstm(subwords, (torch.cat(h, 0), torch.cat(c, 0)))

            output = torch.sigmoid(self.compose(torch.cat([words_out, char_context[idx]], 1)))

            self._take_action(output, h, c, insts, golds, idx, char_idx_list)
            pred_list.append(output.unsqueeze(1))
        return torch.cat(pred_list, 1)

    def _take_action(self, output, h, c, insts, golds, idx, char_idx_list):
        for inst_i in range(len(char_idx_list)):
            if self.training:
                action = golds[inst_i][idx] if idx < len(golds[inst_i]) else Constants.actionPadId
            else:
                action = output[inst_i].argmax()
            if action == Constants.SEP:
                word = ''
                for char_i in range(char_idx_list[inst_i], idx):
                    word += self.id2char[insts[inst_i][char_i]]
                word = torch.tensor([self.word2id[word] if word in self.word2id else Constants.oovId])
                # if self.use_cuda:
                #     word.to(self.device)
                word = self.word_embed(word)
                h_temp, c_temp = self.lstm(word, (h[inst_i], c[inst_i]))
                h[inst_i], c[inst_i] = h_temp, c_temp
                char_idx_list[inst_i] = idx

    def _compose_subword(self, idx, char_idx_list, char_context):
        subword_list = []
        for inst_i in range(len(char_idx_list)):
            if char_idx_list[inst_i] < idx:
                subword = char_context[char_idx_list[inst_i]:idx, inst_i:inst_i+1, :]
                subword = F.avg_pool1d(subword.reshape((1, 1, -1)), subword.shape[0]).squeeze(0)
            else:
                subword = torch.tensor([[0.]*self.char_lstm_hid_dim*2])
            subword_list.append(subword)
        subwords = torch.cat(subword_list, 0)
        return self.char_compose(subwords)

