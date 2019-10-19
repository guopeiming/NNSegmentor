# @Author : guopeiming
# @Datetime : 2019/10/11 17:40
# @File : config.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
from config import Constants
from configparser import ConfigParser


class MyConf(ConfigParser):
    """
    MyConf
    """
    def __init__(self, config_file, *args, **kwargs):
        super(MyConf, self).__init__(*args, **kwargs)
        self.read(filenames=config_file, encoding='utf-8')

    def __str__(self):
        res = ""
        for sect in self.sections():
            res = res + '[' + sect + ']\n'
            for k, v in self.items(sect):
                res = res + k + ' : ' + v + '\n'
            res += '\n'
        return res

    @property
    def char_min_fre(self):
        return self.getint('Preprocess', 'char_min_fre')

    @property
    def word_min_fre(self):
        return self.getint('Preprocess', 'word_min_fre')

    @property
    def data_path(self):
        return self.get('Data', 'data_path')

    @property
    def pretrained_embed_char(self):
        return self.getboolean('Embed', 'pretrained_embed_char')

    @property
    def pretrained_embed_word(self):
        return self.getboolean('Embed', 'pretrained_embed_word')

    @property
    def char_gen_oov_uniform(self):
        return self.getfloat('Embed', 'char_gen_oov_uniform')

    @property
    def word_gen_oov_uniform(self):
        return self.getfloat('Embed', 'word_gen_oov_uniform')

    @property
    def pretrained_char_embed_file(self):
        return self.get('Embed', 'pretrained_char_embed_file')

    @property
    def pretrained_word_embed_file(self):
        return self.get('Embed', 'pretrained_word_embed_file')

    @property
    def char_gen_oov_mode(self):
        return self.get('Embed', 'char_gen_oov_mode')

    @property
    def word_gen_oov_mode(self):
        return self.get('Embed', 'word_gen_oov_mode')

    @property
    def use_cuda(self):
        return self.getboolean('Train', 'use_cuda')

    @property
    def batch_size(self):
        return self.getint('Train', 'batch_size')

    @property
    def shuffle(self):
        return self.getboolean('Train', 'shuffle')

    @property
    def num_workers(self):
        return self.getint('Train', 'num_workers')

    @property
    def drop_last(self):
        return self.getboolean('Train', 'drop_last')

    @property
    def epoch(self):
        return self.getint('Train', 'epoch')

    @property
    def logInterval(self):
        return self.getint('Train', 'logInterval')

    @property
    def valInterval(self):
        return self.getint('Train', 'valInterval')

    @property
    def saveInterval(self):
        return self.getint('Train', 'saveInterval')

    @property
    def fine_tune(self):
        return self.getboolean('Train', 'fine_tune')

    @property
    def save_path(self):
        return self.get('Train', 'save_path')

    @property
    def visual__logger_path(self):
        return self.get('Train', 'visual_logger_path')

    @property
    def char_embed_dim(self):
        return self.getint('Model', 'char_embed_dim')

    @property
    def word_embed_dim(self):
        return self.getint('Model', 'word_embed_dim')

    @property
    def char_embed_max_norm(self):
        num = self.getfloat('Model', 'char_embed_max_norm')
        assert num >= 0., 'Char_max_norm must greater than 0.0'
        return None if abs(num - 0.) < Constants.EPSILON else num

    @property
    def word_embed_max_norm(self):
        num = self.getfloat('Model', 'word_embed_max_norm')
        assert num >= 0., 'Word_max_norm must greater than 0.0'
        return None if abs(num - 0.) < Constants.EPSILON else num

    @property
    def char_lstm_hid_dim(self):
        return self.getint('Model', 'char_lstm_hid_dim')

    @property
    def char_lstm_layers(self):
        return self.getint('Model', 'char_lstm_layers')

    @property
    def word_lstm_hid_dim(self):
        return self.getint('Model', 'word_lstm_hid_dim')

    @property
    def word_lstm_layers(self):
        return self.getint('Model', 'word_lstm_layers')

    @property
    def opti_name(self):
        return self.get('Optimizer', 'name')

