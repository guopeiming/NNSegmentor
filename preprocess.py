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


def read_file(dic, filename):
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip().replace(' ', ''))
            for char in line:
                dic[char] = dic[char]+1 if char in dic else 1


def build_vocab(word2id, id2word, args, config):
    print("Start to build vocab...")
    id_ = 1
    dic = {}
    read_file(dic, args.train)
    read_file(dic, args.dev)
    read_file(dic, args.test)
    for char in dic:
        if dic[char] >= config.min_fre:
            word2id[char] = id_
            id2word.append(char)
            id_ += 1
    word2id[config.padKey] = len(id2word)
    id2word.append(config.padKey)
    assert id_+1 == len(word2id) and id_+1 == len(id2word), "Building vocab goes wrong"
    print("Building vocab completes, length of vocab is %d." % len(word2id))


def convert_insts(filename, word2id, type_, config):
    print("Start to convert %s text data..." % type_)
    assert config.SEP == 1 and config.APP == 0, "SEP and APP can not be changed."
    insts = []
    golds = []
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip())
            if len(line) <= 0: continue

            inst = [word2id[line[0]] if line[0] in word2id else config.oovId]
            gold = [config.SEP]
            for i in range(1, len(line)):
                if line[i] is not ' ':
                    inst.append(word2id[line[i]] if line[i] in word2id else config.oovId)
                    gold.append(config.SEP if line[i-1] is ' ' else config.APP)
            insts.append(inst)
            golds.append(gold)
    assert len(insts) == len(golds), "Converting %s text data goes wrong" % type_
    print("Converting %s text data completes, which length is %d." % (type_, len(insts)))
    return {"insts": insts, "golds": golds}


def make_dataset(args, config):
    assert config.oovId == 0, "oovId can not be changed."
    assert config.oovKey == '-oov-', "oovKey can not be changed."
    assert config.padKey == '-pad-', "padKey can not be changed."
    word2id = {config.oovKey: config.oovId}
    id2word = [config.oovKey]
    data = {}
    build_vocab(word2id, id2word, args, config)
    data["train"] = convert_insts(args.train, word2id, "train", config)
    data["dev"] = convert_insts(args.dev, word2id, "dev", config)
    data["test"] = convert_insts(args.test, word2id, "test", config)
    return {"word2id": word2id, "id2word": id2word, "data": data}


def main():
    args, config = parse_args()
    dataset = make_dataset(args, config)
    torch.save(dataset, args.output_file)
    print("Output file is saved at %s." % args.output_file)


if __name__ == '__main__':
    print("Preprocess starts...")
    main()
    print("Preprocess ends.")

