# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
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
    def bichar_min_fre(self):
        return self.getint('Preprocess', 'bichar_min_fre')

    @property
    def data_path(self):
        return self.get('Data', 'data_path')

    @property
    def dict_data_path(self):
        return self.get('Data', 'dict_data_path')

    @property
    def logits_path(self):
        return self.get('Data', 'logits_path')

    @property
    def cache_3gram_path(self):
        return self.get('Data', 'cache_3gram_path')

    @property
    def pretrained_embed_char(self):
        return self.getboolean('Embed', 'pretrained_embed_char')

    @property
    def pretrained_embed_bichar(self):
        return self.getboolean('Embed', 'pretrained_embed_bichar')

    @property
    def char_gen_oov_uniform(self):
        return self.getfloat('Embed', 'char_gen_oov_uniform')

    @property
    def bichar_gen_oov_uniform(self):
        return self.getfloat('Embed', 'bichar_gen_oov_uniform')

    @property
    def pretrained_char_embed_file(self):
        return self.get('Embed', 'pretrained_char_embed_file')

    @property
    def pretrained_bichar_embed_file(self):
        return self.get('Embed', 'pretrained_bichar_embed_file')

    @property
    def char_gen_oov_mode(self):
        return self.get('Embed', 'char_gen_oov_mode')

    @property
    def bichar_gen_oov_mode(self):
        return self.get('Embed', 'bichar_gen_oov_mode')

    @property
    def seed(self):
        return self.getint('Train', 'seed')

    @property
    def use_cuda(self):
        return self.getboolean('Train', 'use_cuda')

    @property
    def cuda_id(self):
        return self.getint('Train', 'cuda_id')

    @property
    def distillLambda(self):
        return self.getfloat('Train', 'distillLambda')

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
    def accumulation_steps(self):
        return self.getint('Train', 'accumulation_steps')

    @property
    def logInterval(self):
        return self.getint('Train', 'logInterval')

    @property
    def valInterval(self):
        return self.getint('Train', 'valInterval')

    @property
    def visuParaInterval(self):
        return self.getint('Train', 'visuParaInterval')

    @property
    def saveInterval(self):
        return self.getint('Train', 'saveInterval')

    @property
    def save_path(self):
        return self.get('Train', 'save_path')

    @property
    def visual_logger_path(self):
        return self.get('Train', 'visual_logger_path')
    @property
    def is_stack(self):
        return self.getboolean('Model', 'is_stack')

    @property
    def char_embed_dim(self):
        return self.getint('Model', 'char_embed_dim')

    @property
    def char_embed_dim_no_static(self):
        return self.getint('Model', 'char_embed_dim_no_static')

    @property
    def bichar_embed_dim(self):
        return self.getint('Model', 'bichar_embed_dim')

    @property
    def bichar_embed_dim_no_static(self):
        return self.getint('Model', 'bichar_embed_dim_no_static')

    @property
    def dropout_embed(self):
        return self.getfloat('Model', 'dropout_embed')

    @property
    def char_embed_max_norm(self):
        num = self.getfloat('Model', 'char_embed_max_norm')
        assert num >= 0., 'Char_max_norm must greater than 0.0'
        return None if abs(num - 0.) < Constants.EPSILON else num

    @property
    def bichar_embed_max_norm(self):
        num = self.getfloat('Model', 'bichar_embed_max_norm')
        assert num >= 0., 'Word_max_norm must greater than 0.0'
        return None if abs(num - 0.) < Constants.EPSILON else num

    @property
    def encoder_embed_dim(self):
        return self.getint('Model', 'encoder_embed_dim')

    @property
    def dropout_encoder_embed(self):
        return self.getfloat('Model', 'dropout_encoder_embed')

    @property
    def encoder_lstm_hid_size(self):
        return self.getint('Model', 'encoder_lstm_hid_size')

    @property
    def dropout_encoder_hid(self):
        return self.getfloat('Model', 'dropout_encoder_hid')

    @property
    def subword_lstm_hid_size(self):
        return self.getint('Model', 'subword_lstm_hid_size')

    @property
    def word_lstm_hid_size(self):
        return self.getint('Model', 'word_lstm_hid_size')

    @property
    def freeze_bert(self):
        return self.getboolean('Model', 'freeze_bert')

    @property
    def freeze_bert_layers(self):
        return self.getint('Model', 'freeze_bert_layers')

    @property
    def opti_name(self):
        return self.get('Optimizer', 'name')

    @property
    def learning_rate(self):
        return self.getfloat('Optimizer', 'learning_rate')

    @property
    def fine_tune_lr(self):
        return self.getfloat('Optimizer', 'fine_tune_lr')

    @property
    def weight_decay(self):
        return self.getfloat('Optimizer', 'weight_decay')

    @property
    def clip_grad(self):
        return self.getboolean('Optimizer', 'clip_grad')

    @property
    def clip_grad_max_norm(self):
        return self.getfloat('Optimizer', 'clip_grad_max_norm')

    @property
    def warmup_steps(self):
        return self.getint('Optimizer', 'warmup_steps')

    @property
    def lr_decay_factor(self):
        return self.getfloat('Optimizer', 'lr_decay_factor')

    @property
    def momentum(self):
        return self.getfloat('Optimizer', 'momentum')

    @property
    def dampening(self):
        return self.getfloat('Optimizer', 'dampening')

    @property
    def nesterov(self):
        return self.getboolean('Optimizer', 'nesterov')

