# @Author : guopeiming
# @Contact : guopeiming2016@{qq, gmail, 163}.com
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
from config import Constants
from utils.optim import Optim
from config.config import MyConf
from utils.bert_utils import load_data
from utils.visualLogger import VisualLogger
from model.Bert3Gram import Bert3Gram


def parse_args():
    parser = argparse.ArgumentParser(description="NNTranSegmentor")
    parser.add_argument("--config", dest="config", type=str, required=False, default="./config/config.cfg",
                        help="path of config_file")
    args = parser.parse_args()
    config = MyConf(args.config)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cal_preformance(pred, golds, criterion, device):
    batch_size, seq_len = golds.shape[0], golds.shape[1]
    golds_last_seg_idx = torch.tensor([seq_len]*batch_size, dtype=torch.long).to(device)
    pred_last_seg_idx = torch.tensor([seq_len]*batch_size, dtype=torch.long).to(device)
    pred_label = torch.argmax(pred, 2)
    seg_word, char, cor_char, pred_word, golds_word = 0, 0, 0, 0, 0
    for idx in range(seq_len-1, -1, -1):
        pred_seg_mask = pred_label[:, idx] == Constants.SEG
        golds_seg_mask = golds[:, idx] == Constants.SEG
        no_pad_mask = golds[:, idx] != Constants.actionPadId
        seg_word += torch.sum((pred_last_seg_idx == golds_last_seg_idx)[pred_seg_mask * golds_seg_mask]).item()
        pred_last_seg_idx[pred_seg_mask*no_pad_mask] = idx
        golds_last_seg_idx[golds_seg_mask*no_pad_mask] = idx

        char += torch.sum(no_pad_mask).item()
        cor_char += torch.sum((golds[:, idx] - pred_label[:, idx]) == 0).item()
        pred_word += torch.sum(pred_label[:, idx][no_pad_mask] == Constants.SEG).item()
        golds_word += torch.sum(golds[:, idx] == Constants.SEG).item()
        assert seg_word <= pred_word and seg_word <= golds_word, 'evaluation criteria wrong.'

    mask = golds != Constants.actionPadId
    pred, golds = torch.masked_select(pred, mask.unsqueeze(2)).view(-1, 2), torch.masked_select(golds, mask)  # [seq_len_sum, 2/1]
    assert golds.shape[0] == pred.shape[0], 'golds and pred\'s shape are different.'

    loss = criterion(pred, golds)

    return loss, golds_word, pred_word, seg_word, char, cor_char


@torch.no_grad()
def eval_model(model, criterion, dev_data, test_data, device, visual_logger, stamp):
    model.eval()
    print('Validation starts...')
    F_dev = eval_dataset(model, criterion, dev_data, device, 'dev', visual_logger, stamp)
    F_test = eval_dataset(model, criterion, test_data, device, 'test', visual_logger, stamp)
    print('Validation ends.')
    return F_dev, F_test


def eval_dataset(model, criterion, data, device, typ, visual_logger, stamp):
    total_loss, golds_words, pred_words, seg_words, chars, cor_chars = 0.0, 0, 0, 0, 0, 0
    for insts, golds in data:
        golds = golds.to(device)

        pred = model(insts, golds)
        loss, golds_word, pred_word, seg_word, char, cor_char = cal_preformance(pred, golds, criterion, device)
        total_loss += loss.item()
        golds_words += golds_word
        pred_words += pred_word
        seg_words += seg_word
        chars += char
        cor_chars += cor_char
    avg_loss = total_loss/chars
    P = seg_words/pred_words
    R = seg_words/golds_words
    F = (2*P*R)/(P+R)
    print('Model performance in %s dataset Loss: %.05f, F: %.05f, P: %.05f, R: %.05f' %
          (typ, avg_loss, F, P, R))
    scal = {'Loss': avg_loss, 'F': F, 'P': P, 'R': R}
    visual_logger.visual_scalars(scal, stamp, typ)
    return F


def main():
    config = parse_args()
    set_seed(config.seed)

    # ========= Loading Dataset ========= #
    print(config)
    print("Loading dataset starts...")
    train_data, dev_data, test_data, train_dataset = load_data(config)
    print('\n\n', end='')

    # ========= Preparing Model ========= #
    print("Preparing Model starts...")
    if config.use_cuda and torch.cuda.is_available():
        config.device = torch.device('cuda:'+str(config.cuda_id))
        print('You will train model in cuda: %d.\n' % config.device.index)
    else:
        config.device = torch.device('cpu')
        print('GPU is not available, use CPU default.\n')

    model = Bert3Gram(config.device, config.cache_3gram_path)
    if config.use_cuda and torch.cuda.is_available():
        model.to(config.device)
    print(model, end='\n\n\n')

    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(config.device)
    optimizer = Optim(model, config)
    visual_logger = VisualLogger(config.visual_logger_path)

    # ========= Training ========= #
    print('Training starts...')
    start = time.time()
    total_loss, golds_words, pred_words, seg_words, chars, cor_chars, steps = 0.0, 0, 0, 0, 0, 0, 1
    best_perf = [0, 0, 0., 0.]  # (epoch_idx, batch_idx, F_dev, F_test)
    if config.freeze_bert:
        optimizer.set_freeze_by_idxs([str(num) for num in range(0, config.freeze_bert_layers)], True)
        optimizer.free_embeddings()
        optimizer.freeze_pooler()
    for epoch_i in range(config.epoch):
        for batch_i, [insts, golds] in enumerate(train_data):
            golds = golds.to(config.device)
            model.train()

            pred = model(insts, golds)
            loss, golds_word, pred_word, seg_word, char, cor_char = cal_preformance(pred, golds, criterion, config.device)
            total_loss += loss.item()
            golds_words += golds_word
            pred_words += pred_word
            seg_words += seg_word
            chars += char
            cor_chars += cor_char

            loss.backward()

            if steps % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            if steps % config.logInterval == 0:
                avg_loss = total_loss/chars
                P = seg_words/pred_words
                R = seg_words/golds_words
                F = (2*P*R)/(P+R)
                print('[%d/%d], [%d/%d] Loss: %.05f, F: %.05f, P: %.05f, R: %.05f' %
                      (epoch_i+1, config.epoch, batch_i+1, len(train_data), avg_loss, F, P, R))
                sys.stdout.flush()
                scal = {'Loss': avg_loss, 'F': F, 'P': P, 'R': R, 'lr': optimizer.get_lr()[0]}
                visual_logger.visual_scalars(scal, steps, 'train')
                total_loss, golds_words, pred_words, seg_words, chars, cor_chars = 0.0, 0, 0, 0, 0, 0
                # break
            if steps % config.valInterval == 0:
                F_dev, F_test = eval_model(model, criterion, dev_data, test_data, config.device, visual_logger, steps)
                if F_dev > best_perf[2]:
                    best_perf[0], best_perf[1], best_perf[2], best_perf[3] = epoch_i+1, batch_i+1, F_dev, F_test
                print('best performance: [%d/%d], [%d/%d], F_dev: %.05f, F_test: %.05f.' %
                      (best_perf[0], config.epoch, best_perf[1], len(train_data), best_perf[2], best_perf[3]))
                sys.stdout.flush()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                # torch.save(model.pack_state_dict(), os.path.join(config.save_path, 'cnn.pt'))
            if steps % config.visuParaInterval == 1:
                visual_logger.visual_histogram(model, steps)
            if steps % config.saveInterval == 0:
                if not os.path.exists(config.save_path):
                    os.mkdir(config.save_path)
                filename = '%d.model' % steps
                modelpath = os.path.join(config.save_path, filename)
                torch.save(model, modelpath)
            steps += 1
    exe_time = time.time() - start
    print('Executing time: %dh:%dm:%ds.' % (exe_time/3600, (exe_time/60) % 60, exe_time % 60))
    visual_logger.close()
    print('Training ends.')


if __name__ == '__main__':
    main()

