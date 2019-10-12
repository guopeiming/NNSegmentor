# @Author : guopeiming
# @Datetime : 2019/10/11 16:48
# @File : preprocess.py
# @Last Modify Time : 2019/10/11 16:48
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import torch
import argparse
import unicodedata
from config.config import MyConf


def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument("-o", "--output", dest="output_file", type=str, required=True, help="config_file path")
    parser.add_argument("--train", dest="train", type=str, required=True, help="path of train text")
    parser.add_argument("--dev", dest="dev", type=str, help="path of dev text")
    parser.add_argument("--test", dest="test", type=str, required=True, help="path of test text")
    parser.add_argument("--min_fre", type=int, required=True, help="the smallest word frequency in the vocab")
    args = parser.parse_args()

    config = MyConf("./config/config.cfg")
    return args, config


def build_vocab(word2id, id2word, args):
    print("Start to build vocab...")
    id_ = 1
    dic = {}
    with open(args.train, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip().replace(' ', ''))
            for char in line:
                dic[char] = dic[char]+1 if char in dic else 0
    for char in dic:
        if dic[char] >= args.min_fre:
            word2id[char] = id_
            id2word.append(char)
            id_ += 1
    assert id_ == len(word2id) and id_ == len(id2word), "Building vocab goes wrong"
    print("Building vocab completes, length of vocab is %d." % len(word2id))


def convert_insts(filename, word2id, type_, config):
    print("Start to convert %s text data..." % type_)
    insts = []
    golds = []
    with open(filename, mode="r", encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = unicodedata.normalize("NFKC", line.strip())
            if len(line) <= 0: continue

            # if len(line) is 0: print(num)
            inst = [word2id[line[0]] if line[0] in word2id else config.OOVid]
            gold = [config.SEP]
            for i in range(1, len(line)):
                if line[i] is not ' ':
                    inst.append(word2id[line[i]] if line[i] in word2id else config.OOVid)
                    gold.append(config.SEP if line[i-1] is ' ' else config.APP)
            insts.append(inst)
            golds.append(gold)
    assert len(insts) == len(golds), "Converting %s text data goes wrong" % type_
    print("Converting %s text data completes, which length is %d." % (type_, len(insts)))
    return {"insts": insts, "golds": golds}


def make_dataset(args, config):
    word2id = {"OOVid": 0}
    id2word = ["OOVid"]
    data = {}
    build_vocab(word2id, id2word, args)
    data["train"] = convert_insts(args.train, word2id, "train", config)
    if args.dev is not None:
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

