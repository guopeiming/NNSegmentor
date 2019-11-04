# @Author : guopeiming
# @Datetime : 2019/11/03 21:07
# @File : dataset.py
# @Last Modify Time : 2019/11/03 21:07
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import torch.nn as nn
import torch.nn.init as init


class SubwordStackLSTMCell(nn.Module):
    """
    SubwordStackLSTMCell class
    """
    def __init__(self, input_size, hidden_size, device):
        super(SubwordStackLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.batch_idx = None
        self.pos_word = None
        self.pos_subword = None
        self.stack_hidden = None
        self.stack_cell = None
        self.lstm_r = nn.LSTMCell(self.input_size, self.hidden_size, bias=True)
        self.lstm_l = nn.LSTMCell(self.input_size, self.hidden_size, bias=True)
        self.word_compose = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
        self.__init_para()

    def init_stack(self, max_words_num, max_subword_size, batch_size):
        """
        init the stack
        :param max_words_num: int, equals seq_len, means the max number of words that a inst can be segmented.
        :param max_subword_size: int, equals seq_len+1, means the max size of a subword.
        :param batch_size: int, the number of inst for a batch.
        :return:
        """
        self.stack_hidden = torch.zeros((batch_size, max_words_num, max_subword_size, self.hidden_size)).to(self.device)
        self.stack_cell = torch.zeros((batch_size, max_words_num, max_subword_size, self.hidden_size)).to(self.device)
        self.batch_idx = torch.arange(batch_size, dtype=torch.long).to(self.device)
        self.pos_word = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        self.pos_subword = torch.zeros(batch_size, dtype=torch.long).to(self.device)

    def forward(self, char):
        """
        generate subword/word representation by biLSTM and non-linear layer.
        :param char: (batch_size, self.input_size) output of char_encoder.
        :return: subword/word representation
        """
        h = self.stack_hidden[self.batch_idx, self.pos_word, self.pos_subword, :]
        c = self.stack_cell[self.batch_idx, self.pos_word, self.pos_subword, :]
        h, c = self.lstm_r(char, (h, c))
        self.stack_hidden[self.batch_idx, self.pos_word, self.pos_subword+1, :] = h
        self.stack_cell[self.batch_idx, self.pos_word, self.pos_subword+1, :] = c
        h_l, c_l = self.lstm_l(char)
        subword = self.word_compose(torch.cat([h, h_l], 1))
        return subword

    def update_pos(self, op_word, op_subword):
        """
        update the self.pos_word and self.pos_char depending on operation.
        :param op_word: (batch_size, ) -1 means pop, 0 means hold, 1 means push.
        :param op_subword: (batch_size, ) -1 means pop, 0 means hold, 1 means push.
        :return:
        """
        self.pos_word = self.pos_word + op_word
        self.pos_subword = self.pos_char + op_subword

    def __init_para(self):
        init.xavier_uniform_(self.lstm_l.weight_hh)
        init.xavier_uniform_(self.lstm_l.weight_ih)
        init.uniform_(self.lstm_l.bias_hh)
        init.uniform_(self.lstm_l.bias_ih)
        init.xavier_uniform_(self.lstm_r.weight_hh)
        init.xavier_uniform_(self.lstm_r.weight_ih)
        init.uniform_(self.lstm_l.bias_hh)
        init.uniform_(self.lstm_l.bias_ih)
        init.xavier_uniform_(self.word_compose[0].weight)
        init.uniform_(self.word_compose[0].bias)


if __name__ == '__main__':
    model = SubwordStackLSTMCell(3, 3, torch.device('cpu'))
    print(model)

