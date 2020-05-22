import time
import torch
import numpy as np
import unicodedata
from typing import List, Set
from utils.bert_utils import CWSBertDataset, pad_collate_fn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from model.testModel import BertMdoel_test, Bert3Gram_test


def fun(filename: str, dic: Set) -> List[List[List[int]]]:
    res = []
    with open(filename, encoding='utf-8') as reader:
        for line in reader:
            inst = []
            line = '--' + unicodedata.normalize('NFKC', line.strip()).replace(' ', '') + '--'
            for i in range(len(line)-4):
                data = []
                for j in range(5):
                    for k in range(j, 5):
                        data.append(1 if line[i+j:i+k+1] in dic else 0)
                assert len(data) == 15
                inst.append(data)
            res.append(inst)
    return res


def get_insts_lsit(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as file:
        for inst in file:
            inst = unicodedata.normalize('NFKC', inst).strip().replace(' ', '')
            res.append(' '.join([char for char in inst]))
    return res


def get_golds(filename: str):
    res = []
    with open(filename, encoding='utf-8') as file:
        for line in file:
            line = unicodedata.normalize('NFKC', line.strip())
            inst = [1]
            for i in range(1, len(line)):
                if line[i] is ' ': continue
                inst.append(1 if line[i-1] is ' ' else 0)
            res.append(inst)
    return res


# if __name__ == '__main__':  # 生成insts和golds数据集
#     data = {
#             'train_insts': get_insts_lsit('./data/weibo/nlpcc2016-word-seg-train.dat'),
#             'dev_insts': get_insts_lsit('./data/weibo/nlpcc2016-wordseg-dev.dat'),
#             'test_insts': get_insts_lsit('./data/weibo/nlpcc2016-wordseg-dev.dat'),
#             'train_golds': get_golds('./data/weibo/nlpcc2016-word-seg-train.dat'),
#             'dev_golds': get_golds('./data/weibo/nlpcc2016-wordseg-dev.dat'),
#             'test_golds': get_golds('./data/weibo/nlpcc2016-wordseg-dev.dat'),
#             }
#     assert len(data['train_golds']) == len(data['train_insts']), print(len(data['golds']))
#     torch.save(data, './data/weibo/weibo.bert.pt')
#     data = torch.load('./data/weibo/weibo.bert.pt')
#     print(data['train_insts'][1012])
#     print(data['train_golds'][1012])
#     print(len(data['train_insts'][0]))
#     print(len(data['train_golds'][0]))
#     print(data['train_insts'][19522])

# if __name__ == '__main__':  # 计算OOV的命中率
#     dict_3gram = dict()
#     dict_3num = dict()
#     with open('./data/weibo/nlpcc2016-word-seg-train.dat', encoding='utf-8') as file:
#         for line in file:
#             line = '-' + unicodedata.normalize('NFKC', line.strip().replace(' ', '')) + '-'
#             for i in range(len(line) - 2):
#                 dict_3gram[line[i: i+3]] = dict_3gram.get(line[i: i+3], 0) + 1
#     print(len(dict_3gram))
#
#     for k, v in dict_3gram.items():
#         dict_3num[v] = dict_3num.get(v, 0) + 1
#     print(dict_3num)
#     print(dict_3num[1])
#     print(dict_3num[2])
#
#     torch.save(set(dict_3gram.keys()), './data/weibo/weibo.3gram.pt')

#     hit = 0
#     test_sum = 0
#     with open('./data/ctb60/dev.ctb60.hwc.seg', encoding='utf-8') as file:
#         for line in file:
#             line = '-' + unicodedata.normalize('NFKC', line.strip().replace(' ', '')) + '-'
#             for i in range(len(line) - 2):
#                 if line[i: i+3] in dict_3gram:
#                     hit += 1
#                 test_sum += 1
#     print(hit/test_sum)
#     print(hit, test_sum)
#
#     hit = 0
#     test_sum = 0
#     with open('./data/ctb60/test.ctb60.hwc.seg', encoding='utf-8') as file:
#         for line in file:
#             line = '-' + unicodedata.normalize('NFKC', line.strip().replace(' ', '')) + '-'
#             for i in range(len(line) - 2):
#                 if line[i: i + 3] in dict_3gram:
#                     hit += 1
#                 test_sum += 1
#     print(hit / test_sum)
#     print(hit, test_sum)


# def load_data() -> DataLoader:
#     data = torch.load('./data/ctb60/bert.data.bin')
#     li = []
#     for inst in data['test_insts']:
#         li.append(inst.replace(' ', ''))
#     test_data = DataLoader(dataset=CWSBertDataset(li, data['test_golds']), batch_size=16,
#                            shuffle=True, num_workers=4, collate_fn=pad_collate_fn, drop_last=False)
#     return test_data


# if __name__ == '__main__':  # 计算模型时间
    # test_data = load_data()
    # device = torch.device('cuda:0')
    # model = BertMdoel_test(device).to(device)
    # for param in model.parameters():
    #     param.requires_grad_(False)
    # start = time.time()
    # for insts, golds in test_data:
    #     golds = golds.to(device)
    #     model.eval()
    #     with torch.no_grad():
    #         pred = model(insts, golds)
    # print(time.time()-start)
    #
    # test_data = load_data()
    # device = torch.device('cuda:0')
    # embeddings = torch.load('embedd.pt')
    # print(len(embeddings))
    # model = Bert3Gram_test(device, embeddings).to(device)
    # for param in model.parameters():
    #     param.requires_grad_(False)
    # start = time.time()
    # for insts, golds in test_data:
    #     golds = golds.to(device)
    #     model.eval()
    #     with torch.no_grad():
    #         pred = model(insts, golds)
    # print(time.time()-start)
    #
    # import random
    # random.random()


# from config.config import MyConf
# from config import Constants
# from utils.bert_utils import load_data
#
#
# def build_logits(model, data):
#     data_dict = dict()
#     for insts, golds, idxs in data:
#         golds = golds.to(config.device)
#         mask = golds != Constants.actionPadId
#         with torch.no_grad():
#             preds = model(insts, golds)
#         for i in range(len(idxs)):
#             data_dict[idxs[i]] = torch.masked_select(preds[i], mask[i].unsqueeze(1)).cpu().tolist()
#     return data_dict
#
#
# if __name__ == '__main__':  # 生成logits.py
#     model = torch.load('bert_model.pt')
#     config = MyConf('./config/config.cfg')
#     config.device = torch.device('cuda:' + str(config.cuda_id))
#     model = model.to(config.device)
#
#     train_data, dev_data, test_data, train_dataset = load_data(config)
#
#     model.train()
#     train_dict = build_logits(model, train_data)
#     model.eval()
#     dev_dict = build_logits(model, dev_data)
#     model.eval()
#     test_dict = build_logits(model, test_data)
#
#     torch.save({'train_logits': train_dict, 'dev_logits': dev_dict, 'test_logits': test_dict}, 'logits.pt')
#     print(len(train_dict), len(dev_dict), len(test_dict))
#     print(len(train_data), len(dev_data), len(test_data))

# if __name__ == '__main__':  # 生成字典向量 数据集
#     dictn = set()
#     with open('./data/weibo/nlpcc2016-word-seg-train.dat', encoding='utf-8') as file1, open('./data/weibo/nlpcc2016-wordseg-dev.dat', encoding='utf-8') as file2, open('./data/weibo/nlpcc2016-wordseg-dev.dat', encoding='utf-8') as file3:
#         files = [file1, file2, file3]
#         for file in files:
#             for inst in file:
#                 inst = unicodedata.normalize('NFKC', inst.strip()).split(' ')
#                 for word in inst:
#                     dictn.add(word)
#     with open('./data/weibo/weibo.dictionary.txt', encoding='utf-8', mode='w+') as file:
#         for item in dictn:
#             file.write(item+'\n')
#     dictn_data = dict()
#     dictn_data['test'] = fun('./data/weibo/nlpcc2016-wordseg-dev.dat', dictn)
#     dictn_data['dev'] = fun('./data/weibo/nlpcc2016-wordseg-dev.dat', dictn)
#     dictn_data['train'] = fun('./data/weibo/nlpcc2016-word-seg-train.dat', dictn)
#     torch.save(dictn_data, './data/weibo/weibo.dict.data.pt')
#     print(dictn_data['test'][0][0], dictn_data['test'][0][1])
#     print('海浦' in dictn)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     x = [3, 5, 7, 9]
#     y = [96.5, 96.8, 96.9, 96.7]
#     plt.plot(x, y)
#     plt.xlabel('CNN kernel size')
#     plt.ylabel('F1 score')
#     plt.show()



