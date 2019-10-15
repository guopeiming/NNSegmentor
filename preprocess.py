# @Author : guopeiming
# @Datetime : 2019/10/11 16:48
# @File : preprocess.py
# @Last Modify Time : 2019/10/14 20:16
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import argparse
import unicodedata
from config.config import MyConf


def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument("-o", "--output", dest="output_file", type=str, required=True, help="config_file path")
    parser.add_argument("--train", dest="train", type=str, required=True, help="path of train text")
    parser.add_argument("--dev", dest="dev", type=str, required=True, help="path of dev text")
    parser.add_argument("--test", dest="test", type=str, required=True, help="path of test text")
    args = parser.parse_args()

    config = MyConf("./config/config.cfg")
    return args, config


def read_file(char_dic, word_dic, filename):
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip()).split(' ')
            for word in line:
                word_dic[word] = word_dic[word]+1 if word in word_dic else 1
            line = ''.join(line)
            for char in line:
                char_dic[char] = char_dic[char]+1 if char in char_dic else 1


def convert_dic(dic, min_fre, config):
    assert config.oovId == 0, "oovId can not be changed."
    assert config.padId == 1, "padId can not be changed."
    assert config.oovKey == '-oov-', "oovKey can not be changed."
    assert config.padKey == '-pad-', "padKey can not be changed."
    item2id = {config.oovKey: config.oovId}
    id2item = [config.oovKey]
    item2id[config.padKey] = len(id2item)
    id2item.append(config.padKey)
    id_ = 2
    for item in dic:
        if dic[item] >= min_fre:
            item2id[item] = id_
            id2item.append(item)
            id_ += 1
    assert id_ == len(item2id) and id_ == len(id2item), "Building vocab goes wrong"
    return item2id, id2item


def build_vocab(args, config):
    print("Start to build vocab...")
    char_dic = {}
    word_dic = {}
    read_file(char_dic, word_dic, args.train)
    read_file(char_dic, word_dic, args.dev)
    read_file(char_dic, word_dic, args.test)
    char2id, id2char = convert_dic(char_dic, config.char_min_fre, config)
    print("Building %s vocab completes, which is %d." % ('char', len(char2id)))
    word2id, id2word = convert_dic(word_dic, config.word_min_fre, config)
    print("Building %s vocab completes, which is %d." % ('word', len(word2id)))
    return {'char2id': char2id, 'id2char': id2char, 'word2id': word2id, 'id2word': id2word}


def convert_insts(filename, char2id, type_, config):
    print("Start to convert %s text data..." % type_)
    assert config.SEP == 1 and config.APP == 0, "SEP and APP can not be changed."
    insts = []
    golds = []
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip())
            if len(line) <= 0: continue

            inst = [char2id[line[0]] if line[0] in char2id else config.oovId]
            gold = [config.SEP]
            for i in range(1, len(line)):
                if line[i] is not ' ':
                    inst.append(char2id[line[i]] if line[i] in char2id else config.oovId)
                    gold.append(config.SEP if line[i-1] is ' ' else config.APP)
            insts.append(inst)
            golds.append(gold)
    assert len(insts) == len(golds), "Converting %s text data goes wrong" % type_
    print("Converting %s text data completes, which length is %d." % (type_, len(insts)))
    return {"insts": insts, "golds": golds}


def make_dataset(args, config):
    data = {}
    dic = build_vocab(args, config)
    data["train"] = convert_insts(args.train, dic['char2id'], "train", config)
    data["dev"] = convert_insts(args.dev, dic['char2id'], "dev", config)
    data["test"] = convert_insts(args.test, dic['char2id'], "test", config)
    return {'dic': dic, 'data': data}


def test(dataset):
    id2char = dataset["dic"]["id2char"]
    inst = dataset["data"]['train']['insts'][0]
    for id_ in inst:
        print(id2char[id_], end='')
    print()
    print(dataset["data"]['train']['golds'][0])


def main():
    args, config = parse_args()
    dataset = make_dataset(args, config)
    torch.save(dataset, args.output_file)
    print("Output file is saved at %s." % args.output_file)
    # test(dataset)


if __name__ == '__main__':
    print("Preprocess starts...")
    main()
    print("Preprocess ends.")

