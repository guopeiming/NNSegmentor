# @Author : guopeiming
# @Datetime : 2019/10/16 14:17
# @File : dataset.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import numpy as np
import torch.nn.init as init
from utils.data_utils import load_pretrained_char_embed


import torch.nn as nn
from config import Constants


class CharEncoder(nn.Module):
    """
    submodel of NNTransSegmentor ------ CharEncoder
    """
    def __init__(self, char_vocab_size, id2char, config):
        super(CharEncoder, self).__init__()

        if config.pretrained_embed_char:
            embeddings = load_pretrained_char_embed(id2char, config)
            self.embed = nn.Embedding.from_pretrained(embeddings, freeze=config.fine_tune, padding_idx=Constants.padId,
                                                      max_norm=config.char_embed_max_norm)
        else:
            self.embed = nn.Embedding(char_vocab_size, config.char_embed_dim, padding_idx=Constants.padId,
                                      max_norm=config.char_embed_max_norm)
            self.embed.weight.requires_grad_(config.fine_tune)
            init.normal_(self.embed.weight, mean=0.0, std=np.sqrt(5. / config.char_embed_dim))

        self.lstm = nn.LSTM(config.char_embed_dim, config.char_lstm_hid_dim, config.char_lstm_layers,
                            bias=True, dropout=0., bidirectional=True)
        init.normal_(self.lstm.weight_ih, mean=0.0, std=np.sqrt(3. / config.char_embed_dim))
        init.normal_(self.lstm.weight_hh, mean=0.0, std=np.sqrt(3. / config.char_embed_dim))
        init.uniform_(self.lstm.bias_ih)
        init.uniform_(self.lstm.bias_hh)

    def forward(self, insts):
        output, (h_n, c_n) = self.lstm(insts)
        return output

