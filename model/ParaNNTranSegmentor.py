# @Author : guopeiming
# @Datetime : 2019/11/14 20:13
# @File : dataset.py
# @Last Modify Time : 2019/11/14 20:13
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}


import torch
import torch.nn as nn
from config import Constants
from model.char_encoder import CharEncoder
from model.StackLSTMCell import StackLSTMCell
from model.SubwordLSTMCell import SubwordLSTMCell


class ParaNNTranSegmentor(nn.Module):
    """
    ParaNNTranSegmentor
    """

    def __init__(self, pretra_char_embed, char_embed_num, char_embed_dim, char_embed_max_norm, pretra_bichar_embed,
                 bichar_embed_num, bichar_embed_dim, bichar_embed_max_norm, encoder_lstm_hid_size,
                 subword_lstm_hid_size, word_lstm_hid_size, device):
        super(ParaNNTranSegmentor, self).__init__()

        self.char_encoder = CharEncoder(pretra_char_embed, char_embed_num, char_embed_dim, char_embed_max_norm,
                                        pretra_bichar_embed, bichar_embed_num, bichar_embed_dim, bichar_embed_max_norm,
                                        encoder_lstm_hid_size)
        self.subwStackLSTM = SubwordLSTMCell(encoder_lstm_hid_size*2, subword_lstm_hid_size, device)
        self.wordStackLSTM = StackLSTMCell(subword_lstm_hid_size, word_lstm_hid_size, device)
        self.classifier = nn.Linear(word_lstm_hid_size+2*encoder_lstm_hid_size, 2, bias=True)
        self.device = device
        self.subword_action_map = torch.tensor([1, 2, 2]).to(self.device)
        self.word_action_map = torch.tensor([0, 1, 1]).to(self.device)
        self.__init_para()

    def forward(self, insts, golds=None):
        chars = self.char_encoder(insts)  # (seq_len, batch_size, encoder_lstm_hid_size*2)

        batch_size, seq_len = insts.shape[0], insts.shape[1]
        self.subwStackLSTM.init_stack(2*seq_len+2, batch_size)
        self.wordStackLSTM.init_stack(seq_len+1, batch_size)

        pred = []
        for idx in range(seq_len):
            if idx == 0:
                subword = self.subwStackLSTM(torch.zeros((batch_size, chars.shape[2])).to(self.device)).to(self.device)
            else:
                subword = self.subwStackLSTM(chars[idx-1, :, :])  # [batch_size, word_lstm_hid_size]
            word_repre, _ = self.wordStackLSTM(subword)  # [batch_size, word_lstm_hid_size]
            output = self.classifier(torch.cat([word_repre, chars[idx, :, :]], 1))  # [batch_size, 3]

            if self.training:
                subwordOP = self.subword_action_map.index_select(0, golds[:, idx])
                wordOP = self.word_action_map.index_select(0, golds[:, idx])
            else:
                subwordOP = self.subword_action_map.index_select(0, torch.argmax(output, 1))
                wordOP = self.word_action_map.index_select(0, torch.argmax(output, 1))
            self.subwStackLSTM.update_pos(subwordOP)
            self.wordStackLSTM.update_pos(wordOP)

            pred.append(output.unsqueeze(1))
        return torch.cat(pred, 1)  # [batch, char_num, label_num]

    def __init_para(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.uniform_(self.classifier.bias)
