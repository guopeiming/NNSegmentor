# @Author : guopeiming
# @Datetime : 2019/10/11 17:40
# @File : preprocess.py
# @Last Modify Time : 2019/10/11 17:40
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
from configparser import ConfigParser


class MyConf(ConfigParser):
    """
    MyConf
    """
    def __init__(self, config_file, *args, **kwargs):
        super(MyConf, self).__init__(*args, **kwargs)
        self.read(filenames=config_file, encoding='utf-8')

    @property
    def OOVid(self):
        return self.getint('Constants', 'OOVid')

    @property
    def APP(self):
        return self.getint('Constants', 'APP')

    @property
    def SEP(self):
        return self.getint('Constants', 'SEP')

