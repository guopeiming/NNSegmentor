# @Author : guopeiming
# @Datetime : 2019/10/11 16:48
# @File : preprocess.py
# @Last Modify Time : 2019/10/14 20:16
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import argparse
import unicodedata
from config import Constants
from config.config import MyConf


def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument("--train", dest="train", type=str, required=True, help="path of train text")
    parser.add_argument("--dev", dest="dev", type=str, required=True, help="path of dev text")
    parser.add_argument("--test", dest="test", type=str, required=True, help="path of test text")
    args = parser.parse_args()

    config = MyConf("./config/config.cfg")
    return args, config


def read_file(char_dic, bichar_dic, word_dic, filename):
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip()).split(' ')
            for word in line:
                word_dic[word] = word_dic[word]+1 if word in word_dic else 1
            line = ''.join(line)
            for char in line:
                char_dic[char] = char_dic[char]+1 if char in char_dic else 1
            bichar = Constants.BOS + line[0]
            bichar_dic[bichar] = bichar_dic[bichar]+1 if bichar in bichar_dic else 1
            for idx in range(0, len(line)-1, 1):
                bichar = line[idx] + line[idx+1]
                bichar_dic[bichar] = bichar_dic[bichar]+1 if bichar in bichar_dic else 1
            bichar = line[len(line)-1] + Constants.EOS
            bichar_dic[bichar] = bichar_dic[bichar]+1 if bichar in bichar_dic else 1


def convert_dic(dic, min_fre):
    assert Constants.oovId == 0, "oovId can not be changed."
    assert Constants.padId == 1, "padId can not be changed."
    assert Constants.oovKey == '<unk>', "oovKey can not be changed."
    assert Constants.padKey == '<pad>', "padKey can not be changed."
    item2id = {Constants.oovKey: Constants.oovId}
    id2item = [Constants.oovKey]
    item2id[Constants.padKey] = Constants.padId
    id2item.append(Constants.padKey)
    id_ = 2
    drop_list = []
    for item in dic:
        if dic[item] >= min_fre:
            item2id[item] = id_
            id2item.append(item)
            id_ += 1
        else:
            drop_list.append(item)
    assert id_ == len(item2id) and id_ == len(id2item) and id_+len(drop_list) == len(dic)+2, "Building vocab goes wrong"
    print(str(drop_list)+'are abandoned, which are replaced by \'oov\'.')
    print('drop rate is %.05f = %d/%d' % (len(drop_list)/len(dic), len(drop_list), len(dic)))
    return item2id, id2item


def build_vocab(args, config):
    print("Start to build vocab...")
    char_dic = {}
    word_dic = {}
    bichar_dic = {}
    read_file(char_dic, bichar_dic, word_dic, args.train)
    read_file(char_dic, bichar_dic, word_dic, args.dev)
    read_file(char_dic, bichar_dic, word_dic, args.test)
    char2id, id2char = convert_dic(char_dic, config.char_min_fre)
    print("Building char vocab completes, which is %d." % len(char2id))
    bichar2id, id2bichar = convert_dic(bichar_dic, config.bichar_min_fre)
    print('Building bichar vocab completes, which is %d.' % len(bichar2id))
    word2id, id2word = convert_dic(word_dic, config.word_min_fre)
    print('Building word vocab completes, which is %d.' % len(word2id))
    return {'char2id': char2id, 'id2char': id2char,
            'bichar2id': bichar2id, 'id2bichar': id2bichar,
            'word2id': word2id, 'id2word': id2word}


def expand(inst, vocab, item):
    if item in vocab:
        inst.append(vocab[item])
        res = 0
    else:
        inst.append(Constants.oovId)
        res = 1
    return res


def convert_insts(filename, char2id, bichar2id, type_):
    print("Start to convert %s text data..." % type_)
    assert Constants.SEP == 1 and Constants.APP == 0, "SEP and APP can not be changed."
    insts_char = []
    insts_bichar_l = []
    insts_bichar_r = []
    golds = []
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip())
            if len(line) <= 0: continue

            inst_char = []
            inst_bichar_l = []
            inst_bichar_r = []
            char_oov_num, bichar_l_oov_num, bichar_r_oov_num = 0, 0, 0
            gold = [Constants.SEP]
            for i in range(1, len(line)):
                if line[i] is not ' ':
                    gold.append(Constants.SEP if line[i-1] is ' ' else Constants.APP)
            golds.append(gold)

            line = line.replace(' ', '')
            for i in range(0, len(line)):
                char_oov_num += expand(inst_char, char2id, line[i])
                tag = Constants.BOS if id == 0 else line[i-1]
                bichar_l_oov_num += expand(inst_bichar_l, bichar2id, tag+line[i])
                tag = Constants.EOS if id == len(line)-1 else line[i+1]
                bichar_r_oov_num += expand(inst_bichar_r, bichar2id, line[i]+tag)
            insts_char.append(inst_char)
            insts_bichar_l.append(inst_bichar_l)
            insts_bichar_r.append(inst_bichar_r)

    assert len(insts_char) == len(golds) \
        and len(insts_char) == len(insts_bichar_l) \
        and len(insts_char) == len(insts_bichar_r), "Converting %s text data goes wrong" % type_
    print("Converting %s text data completes, which length is %d." % (type_, len(insts_char)))
    print('The number of oov in char insts, bichar_l insts and bichar_r insts are %d %d %d.' %
          (char_oov_num, bichar_l_oov_num, bichar_r_oov_num))
    return {'insts_char': insts_char,
            'insts_bichar_l': insts_bichar_l,
            'insts_bichar_r': insts_bichar_r,
            'golds': golds}


def make_dataset(args, config):
    data = {}
    vocab = build_vocab(args, config)
    data["train"] = convert_insts(args.train, vocab['char2id'], vocab['bichar2id'], 'train')
    data["dev"] = convert_insts(args.dev, vocab['char2id'], vocab['bichar2id'], 'dev')
    data["test"] = convert_insts(args.test, vocab['char2id'], vocab['bichar'], 'test')
    return {'dic': vocab, 'data': data}


def test(dataset, tag):
    id2char = dataset["dic"]["id2char"]
    id2bichar = dataset["dic"]["bichar2id"]
    inst = dataset["data"]['train']['insts_char'][tag]
    for id_ in inst:
        print(id2char[id_], end='')
    print()
    inst = dataset["data"]["train"]["insts_bichar_l"][tag]
    for id_ in inst:
        print(id2bichar[id_]+' ', end='')
    print()
    inst = dataset['data']['train']['insts_bichar_r'][tag]
    for id_ in inst:
        print(id2bichar[id_]+' ', end='')
    print()
    print(dataset["data"]['train']['golds'][tag])


def main():
    args, config = parse_args()
    dataset = make_dataset(args, config)
    torch.save(dataset, config.data_path)
    print("Output file is saved at %s." % config.data_path)
    test(dataset, 0)
    test(dataset, 300)


if __name__ == '__main__':
    print("Preprocess starts...")
    main()
    print("Preprocess ends.")

