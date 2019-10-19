# @Author : guopeiming
# @Datetime : 2019/10/12 18:18
# @File : train.py
# @Last Modify Time : 2019/10/18 08:33
# @Contact : 1072671422@qq.com, guopeiming2016@{gmail.com, 163.com}
import os
import torch
import argparse
from config import Constants
from utils.optim import Optim
from config.config import MyConf
from utils.data_utils import load_data
from model.NNTranSegmentor import NNTranSegmentor


def parse_args():
    parser = argparse.ArgumentParser(description="NNTranSegmentor")
    parser.add_argument("--config", dest="config", type=str, required=False, default="./config/config.cfg",
                        help="path of config_file")
    args = parser.parse_args()
    config = MyConf(args.config)
    return config


def cal_preformance(pred, golds, TP, FN, FP, TN):
    for insts_i in range(len(golds)):
        for char_i in range(len(golds[insts_i])):
            if golds[insts_i][char_i] == Constants.APP:
                if pred[insts_i][char_i][Constants.APP] > pred[insts_i][char_i][Constants.SEP]: TN = TN + 1
                else: FP = FP + 1
            else:
                if pred[insts_i][char_i][Constants.SEP] >= pred[insts_i][char_i][Constants.APP]: TP = TP + 1
                else: FN = FN + 1

    max_len = max(len(gold) for gold in golds)
    golds = torch.tensor([gold + [Constants.actionPadId] * (max_len - len(gold)) for gold in golds], dtype=torch.int64)
    assert golds.shape[0] == pred.shape[0] and golds.shape[1] == pred.shape[1], 'golds and pred\'s shape are different.'
    loss = torch.nn.functional.cross_entropy(pred.view((pred.shape[0]*pred.shape[1], 3)),
                                             golds.view((golds.shape[0]*golds.shape[1],)),
                                             ignore_index=Constants.actionPadId,
                                             reduction='sum')
    return loss, TP, FN, FP, TN


def eval_model(model, dev_data, test_data, config):
    model.eval()
    print('Validating starts...')
    eval_dataset(model, dev_data, config, 'dev')
    eval_dataset(model, test_data, config, 'test')


def eval_dataset(model, data, config, type):
    total_loss = 0.0
    TP, FN, FP, TN = 0, 0, 0, 0
    for batch, golds in data:
        if config.use_cuda:
            batch = batch.to(config.device)
            golds = golds.to(config.device)

        pred = model(batch)
        loss, TP, FN, FP, TN = cal_preformance(pred, golds, TP, FN, FP, TN)
        total_loss += loss.item()
    total_loss = total_loss/(TP+FN+FP+TN)
    ACC = (TP+TN)/(TP+FN+FP+TN)
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    print('Model performance in %s dataset Loss: %.05f, ACC: %.05f, P: %.05f, R: %.05f' % (type, total_loss, ACC, P, R))


def main():
    config = parse_args()

    # ========= Loading Dataset ========= #
    print(config)
    print("\n\nLoading dataset starts...")
    train_data, dev_data, test_data, train_dataset = load_data(config)

    # ========= Preparing Model ========= #
    print("Preparing Model starts...")
    model = NNTranSegmentor(train_dataset.get_id2char(),
                            train_dataset.get_word2id(),
                            train_dataset.get_char_vocab_size(),
                            train_dataset.get_word_vocab_size(),
                            config)
    config.device = torch.device('cpu')
    if config.use_cuda:
        assert torch.cuda.is_available(), "Cuda is not available."
        config.device = torch.device('cuda:0')
        print('GPU is ready.')
        model.to(config.device)
        print('model is loaded to GPU: %d\n' % config.device.index)
    print('you will train model in %s.\n' % config.device.type)
    print(model, end='\n\n')

    optimizer = Optim(config.opti_name, config, model)

    # ========= Training ========= #
    print('Training starts...')
    total_loss, TP, FN, FP, TN = 0.0, 0, 0, 0, 0
    for epoch_i in range(config.epoch):
        for batch_i, (batch, golds) in enumerate(train_data):
            if config.use_cuda:
                batch = batch.to(config.device)
                golds = golds.to(config.device)
            model.train()

            optimizer.zero_grad()
            pred = model(batch, golds)
            loss, TP, FN, FP, TN= cal_preformance(pred, golds, TP, FN, FP, TN)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (batch_i+1+epoch_i*(len(train_data))) % config.logInterval == 0:
                total_loss = total_loss/(TP+FN+FP+TN)
                ACC = (TP+TN)/(TP+FN+FP+TN)
                P = TP/(TP+FP)
                R = TP/(TP+FN)
                print('[%d/%d], [%d/%d] Loss: %.05f, ACC: %.05f, P: %.05f, R: %.05f' %
                      (epoch_i, config.epoch, batch_i, len(train_data), total_loss, ACC, P, R))
                total_loss, TP, FN, FP, TN = 0.0, 0, 0, 0, 0
            if (batch_i+1+epoch_i*(len(train_data))) % config.valInterval == 0:
                eval_model(model, dev_data, test_data)
            if (batch_i+1+epoch_i*(len(train_data))) % config.saveInterval == 0:
                if not os.path.exists(config.save_path):
                    os.mkdir(config.save_path)
                filename = '%d.model' % (batch_i+1+epoch_i*len(train_data))
                modelpath = os.path.join(config.save_path, filename)
                torch.save(model, modelpath)
    print('Training ends.')


if __name__ == '__main__':
    main()

