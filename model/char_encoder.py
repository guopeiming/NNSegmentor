# @Author : guopeiming
# @Datetime : 2019/10/16 14:17
# @File : dataset.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import torch.nn as nn
import torch.nn.init as init
from config import Constants


class CharEncoder(nn.Module):
    """
    submodel of NNTransSegmentor ------ CharEncoder
    """
    def __init__(self, pretra_char_embed, char_embed_num, char_embed_dim, char_embed_max_norm, pretra_bichar_embed, bichar_embed_num,
                 bichar_embed_dim, bichar_embed_max_norm, encoder_embed_dim, encoder_lstm_hid_size, device):
        super(CharEncoder, self).__init__()

        assert pretra_char_embed.shape[0] == char_embed_num and \
            pretra_char_embed.shape[1] == char_embed_dim and \
            pretra_bichar_embed.shape[0] == bichar_embed_num and \
            pretra_bichar_embed.shape[1] == bichar_embed_dim, 'pretrained embeddings shape error.'

        self.char_embed_static = nn.Embedding.from_pretrained(pretra_char_embed, True, Constants.padId, char_embed_max_norm)
        self.char_embed_no_static = nn.Embedding(char_embed_num, char_embed_dim, Constants.padId, char_embed_max_norm)

        self.bichar_embed_static = nn.Embedding.from_pretrained(pretra_bichar_embed, True, Constants.padId, bichar_embed_max_norm)
        self.bichar_embed_no_static = nn.Embedding(bichar_embed_num, bichar_embed_dim, Constants.padId, bichar_embed_max_norm)

        self.embed_l = nn.Sequential(
            nn.Linear((char_embed_dim+bichar_embed_dim)*2, encoder_embed_dim, bias=True),
            nn.Tanh()
        )
        self.embed_r = nn.Sequential(
            nn.Linear((char_embed_dim+bichar_embed_dim)*2, encoder_embed_dim, bias=True),
            nn.Tanh()
        )

        self.lstm_l = nn.LSTMCell(encoder_embed_dim, encoder_lstm_hid_size, bias=True)
        self.lstm_r = nn.LSTMCell(encoder_embed_dim, encoder_lstm_hid_size, bias=True)

        self.encoder_lstm_hid_size = encoder_lstm_hid_size
        self.device = device

        self.__init_para()

    def forward(self, insts):
        insts_char, insts_bichar_l, insts_bichar_r = insts[0], insts[1], insts[2]  # (batch_size, seq_len)
        batch_size, seq_len = insts_char.shape[0], insts_char.shape[1]
        char_embeddings = self.char_embed_static(insts_char).permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        char_embeddings_no_static = self.char_embed_no_static(insts_char).permute(1, 0, 2)
        char_embeddings = torch.cat([char_embeddings, char_embeddings_no_static], 2)
        bichar_embeddins_l = self.bichar_embed_static(insts_bichar_l).permute(1, 0, 2)
        bichar_embeddins_l_no_static = self.bichar_embed_no_static(insts_bichar_l).permute(1, 0, 2)
        bichar_embeddins_l = torch.cat([bichar_embeddins_l, bichar_embeddins_l_no_static], 2)
        bichar_embeddins_r = self.bichar_embed_static(insts_bichar_l).permute(1, 0, 2)
        bichar_embeddins_r_no_static = self.bichar_embed_no_static(insts_bichar_l).permute(1, 0, 2)
        bichar_embeddins_r = torch.cat([bichar_embeddins_r, bichar_embeddins_r_no_static], 2)

        h_l, c_l, h_r, c_r = list(map(lambda x: x.squeeze(0).to(self.device), torch.zeros((4, batch_size, self.encoder_lstm_hid_size)).chunk(4, 0)))
        encoder_output = []
        for step in range(seq_len):
            embeddins_l = self.embed_l(torch.cat([char_embeddings[step], bichar_embeddins_l[step]], 1))
            embeddins_r = self.embed_r(torch.cat([char_embeddings[step], bichar_embeddins_r[step]], 1))
            h_l, c_l = self.lstm_l(embeddins_l, (h_l, c_l))  # (batch_size, encoder_lstm_hid_size)
            h_r, c_r = self.lstm_r(embeddins_r, (h_r, c_r))
            encoder_output.append(torch.cat([h_l.unsqueeze(0), h_r.unsqueeze(0)], 2))  # (1, batch_size, encoder_lstm_hid_size*2)
        return torch.cat(encoder_output, 0)  # (seq_len, batch_size, encoder_lstm_hid_size*2)

    def __init_para(self):
        init.xavier_uniform_(self.char_embed_no_static.weight)
        init.xavier_uniform_(self.bichar_embed_no_static.weight)
        init.xavier_uniform_(self.lstm_l.weight_ih)
        init.xavier_uniform_(self.lstm_l.weight_hh)
        init.xavier_uniform_(self.lstm_r.weight_ih)
        init.xavier_uniform_(self.lstm_r.weight_hh)
        init.uniform_(self.lstm_l.bias_ih)
        init.uniform_(self.lstm_l.bias_hh)
        init.uniform_(self.lstm_r.bias_ih)
        init.uniform_(self.lstm_r.bias_hh)
        init.xavier_uniform_(self.embed_l[0].weight)
        init.xavier_uniform_(self.embed_r[0].weight)
        init.uniform_(self.embed_l[0].bias)
        init.uniform_(self.embed_r[0].bias)
        self.char_embed_no_static.weight.requires_grad_(True)
        self.bichar_embed_no_static.weight.requires_grad_(True)

