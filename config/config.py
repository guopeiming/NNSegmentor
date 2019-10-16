# @Author : guopeiming
# @Datetime : 2019/10/11 17:40
# @File : config.py
# @Last Modify Time : 2019/10/16 14:01
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
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
    def batch_size(self):
        return self.getint('Data', 'batch_size')

    @property
    def shuffle(self):
        return self.getboolean('Data', 'shuffle')

    @property
    def num_workers(self):
        return self.getint('Data', 'num_workers')

    @property
    def drop_last(self):
        return self.getboolean('Data', 'drop_last')

    @property
    def fine_tune(self):
        return self.getboolean('Embed', 'fine_tune')

    @property
    def pretrained_char_embed_file(self):
        return self.get('Embed', 'pretrained_char_embed_file')

    @property
    def pretrained_word_embed_file(self):
        return self.get('Embed', 'pretrained_word_embed_file')

    @property
    def use_cuda(self):
        return self.getboolean('Train', 'use_cuda')

