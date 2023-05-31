#!/usr/bin/env python37
# -*- coding: utf-8 -*-


import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from utils import *
from model.function import *
from model.TSRec import *

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import metric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='data/retailrocket_seq/',
                    help='dataset directory path: data/retailrocket_seq/last_fm/tafeng')
# preprocess args
parser.add_argument('--max_seq_len', type=int, default=10, help='max_seq_len')
parser.add_argument('--max_repeat_seq_len', type=int, default=10, help='max_repeat_seq_len')
parser.add_argument('--max_time_interval', type=int, default=256, help='max time interval')
parser.add_argument('--max_index_gap', type=int, default=5870, help='max index gap')
parser.add_argument('--max_period', type=int, default=256, help='max period')
parser.add_argument('--repeat_time_interval_matrix_row', type=int, default=10, help='repeat period matrix row')
parser.add_argument('--repeat_time_interval_matrix_col', type=int, default=10, help='repeat period matrix column')
parser.add_argument('--negative_sample_num', type=int, default=3, help='negative sample num')
parser.add_argument('--negative_sample_test_num', type=int, default=100, help='negative sample test num')

parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_dim', type=int, default=50, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=100, help='the dimension of item embedding')
parser.add_argument('--gru_dim', type=int, default=50, help='the dimension of gru output')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='drop out rate')
parser.add_argument('--epoch', type=int, default=200, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--repeat_matrix', action='store_true', help='add repeat matrix')
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--parameter', action='store_true', help='load parameter')
parser.add_argument('--save_path', default='./dict', help='load model path')
parser.add_argument('--save_output_path', default='./test_res', help='load model path')
parser.add_argument('--load_path', default='./best_dict/', help='load model path')
parser.add_argument('--model', type=str, default='TSRec', help='TSRec')
parser.add_argument('--repeat_loss', action='store_true', help='repeat_loss')
parser.add_argument('--loss_type', type=str, default='ce', help='ce/bpr')
args = parser.parse_args()
print(args)
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_res = open(args.save_output_path + '/' + args.dataset_path.split('/')[-2] + '/embdim_' + str(
    args.embed_dim) + '_max_seq_' + str(args.max_seq_len) + '.txt', 'a', encoding='utf-8')
model_res.write(str(args) + '\n')
model_res.close()


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    init_seed(1234)

    if args.dataset_path.split('/')[-2] == 'retailrocket':
        n_items = 51586
    elif args.dataset_path.split('/')[-2] == 'retailrocket_seq':
        n_items = 51586
        n_users = 217508
        n_index_gap = 3654
    elif args.dataset_path.split('/')[-2] == 'last_fm':
        n_items = 11543
        n_users = 728
        n_index_gap = 14494
    elif args.dataset_path.split('/')[-2] == 'tafeng':
        n_items = 30441
        n_users = 11209
        n_index_gap = 884
    elif args.dataset_path.split('/')[-2] == 'diginetica':
        n_items = 24700
        n_users = 187938
        n_index_gap = 30

    else:
        raise Exception('Unknown Dataset!')

    print('Loading data...')
    train_dataset, test_dataset, repeat_min_interval = load_data(args.dataset_path, args.max_seq_len)
    if args.repeat_matrix:
        repeat_matrix = load_matrix(args.dataset_path)
    else:
        repeat_matrix = []

    train_data = RecDataset(train_dataset, repeat_min_interval, args.max_time_interval, args.max_seq_len, use_seq=True)
    test_data = RecDataset(test_dataset, repeat_min_interval, args.max_time_interval, args.max_seq_len, use_seq=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size // 2, shuffle=False, drop_last=True)

    if args.model == 'TSRec':
        model = TSRec(n_users, n_items, args.embed_dim, args.hidden_dim, args.gru_dim, args.max_seq_len,
                      args.max_repeat_seq_len, args.max_time_interval, n_index_gap,
                      args.max_period, args.negative_sample_num, args.negative_sample_test_num).to(device)
    else:
        raise Exception('Unknown Model!')

    if args.test:
        ckpt = torch.load(args.load_path)
        model.load_state_dict(ckpt)
        hr20, mrr20, ndcg20, hr10, mrr10, ndcg10, hr5, mrr5, ndcg5 = test(args,
                                                                                      test_loader, model, repeat_matrix)
        print(
            ('Test: hr@20: {:.4f}, MRR@20: {:.4f}, NDCG@20: {:.4f}\n ' + \
             'hr@10: {:.4f}, MRR@10: {:.4f}, NDCG@10: {:.4f}\n' + \
             'hr@5: {:.4f}, MRR@5: {:.4f}, NDCG@5: {:.4f} \n').format(hr20, mrr20, ndcg20,
                                                                          hr10, mrr10, ndcg10,
                                                                          hr5, mrr5, ndcg5))
        return
    optimizer = optim.Adam(model.parameters(), args.lr)

    print('Training...')

    train(args, train_loader, test_loader, model, optimizer, repeat_matrix)


def train(args, train_loader, test_loader, model, optimizer, repeat_matrix, log_aggr=100):
    best_hr = 0
    best_ndcg = 0
    for epoch in tqdm(range(args.epoch)):

        model.train()
        sum_epoch_loss = 0
        start = time.time()
        model_res = open(args.save_output_path + '/' + args.dataset_path.split('/')[-2] + '/embdim_' + str(
            args.embed_dim) + '_max_seq_' + str(args.max_seq_len) + '.txt', 'a', encoding='utf-8')
        for i, (uid, tar, neg__lis_items, cur_s, last_s, neg_s, cur_ti, repeat_ti, neg_ti, cur_ig,
                repeat_ig, neg_ig, tar_seq, neg_tar_seq) in tqdm(enumerate(train_loader), total=len(train_loader)):

            neg_items = neg__lis_items.reshape(-1)
            repeat_matrix_batch = torch.from_numpy(np.array([repeat_matrix[i] for i in tar.numpy()]))
            zero_matrix = [[0] * args.repeat_period_matrix_col] * args.repeat_period_matrix_row
            repeat_matrix_batch_neg = torch.from_numpy(np.array([repeat_matrix[i] if i in repeat_matrix else zero_matrix
                                                                 for i in neg_items.numpy()]))
            uid = uid.to(dtype=torch.long, device=device)
            tar = tar.to(dtype=torch.long, device=device)
            neg_items = neg_items.to(dtype=torch.long, device=device)
            repeat_matrix_batch = repeat_matrix_batch.to(dtype=torch.long, device=device)
            repeat_matrix_batch_neg = repeat_matrix_batch_neg.to(dtype=torch.long, device=device)
            tar_seq, neg_tar_seq = (tar_seq.to(dtype=torch.long, device=device),
                                    neg_tar_seq.to(dtype=torch.long, device=device))
            cur_s, last_s, neg_s = (cur_s.to(dtype=torch.long, device=device),
                                    last_s.to(dtype=torch.long, device=device),
                                    neg_s.reshape(neg_s.size(0) * neg_s.size(1),
                                                  neg_s.size(2)).to(dtype=torch.long, device=device))

            cur_ti, repeat_ti, neg_ti = (cur_ti.to(dtype=torch.long, device=device),
                                         repeat_ti.to(dtype=torch.long, device=device),
                                         neg_ti.reshape(neg_ti.size(0) * neg_ti.size(1),
                                                        neg_ti.size(2)).to(dtype=torch.long, device=device))
            cur_ig, repeat_ig, neg_ig = (cur_ig.to(dtype=torch.long, device=device),
                                         repeat_ig.to(dtype=torch.long, device=device),
                                         neg_ig.reshape(neg_ig.size(0) * neg_ig.size(1),
                                                        neg_ig.size(2)).to(dtype=torch.long, device=device))
            neg_uid = uid.tile((args.negative_sample_num,)).reshape(args.negative_sample_num * uid.size(0), -1)
            neg_cur_s = cur_s.tile((args.negative_sample_num,)).reshape(args.negative_sample_num * cur_s.size(0), -1)
            neg_cur_ti = cur_ti.tile((args.negative_sample_num,)).reshape(args.negative_sample_num * cur_ti.size(0), -1)
            neg_cur_ig = cur_ig.tile((args.negative_sample_num,)).reshape(args.negative_sample_num * cur_ig.size(0), -1)
            optimizer.zero_grad()

            if args.model == 'TSRec':
                tar_prediction = model(uid, tar, last_s, cur_s, repeat_ti, cur_ti, repeat_ig, cur_ig,
                                       repeat_matrix_batch)
                neg_prediction = model(neg_uid, neg_items, neg_s, neg_cur_s, neg_ti, neg_cur_ti, neg_ig, neg_cur_ig,
                                       repeat_matrix_batch_neg)
                positive_loss = -torch.mean(
                    torch.log(torch.sigmoid(tar_prediction)))
                negative_loss = -torch.mean(
                    torch.log(1 - torch.sigmoid(neg_prediction)))
            else:
                positive_loss = 0
                negative_loss = 0

            loss = positive_loss + negative_loss
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            sum_epoch_loss += loss_val

            if i % log_aggr == 0:
                print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f)'
                      % (epoch + 1, args.epoch, loss_val, sum_epoch_loss / (i + 1)))

        hr20, mrr20, ndcg20, hr10, mrr10, ndcg10, hr5, mrr5, ndcg5 = test(args,
                                                                                      test_loader, model, repeat_matrix)
        out_res = ('Epoch {} validation: hr@20: {:.4f}, MRR@20: {:.4f}, NDCG@20: {:.4f}\n ' + \
                   'hr@10: {:.4f}, MRR@10: {:.4f}, NDCG@10: {:.4f}\n' + \
                   'hr@5: {:.4f}, MRR@5: {:.4f}, NDCG@5: {:.4f} \n').format(epoch + 1, hr20, mrr20, ndcg20,
                                                                                hr10, mrr10, ndcg10, hr5, mrr5,
                                                                                ndcg5)

        model_res.write(out_res + '\n')
        model_res.close()
        print(out_res)
        if hr20 > best_hr or ndcg20 > best_ndcg:
            if hr20 > best_hr:
                best_hr = hr20
            if ndcg20 > best_ndcg:
                best_ndcg = ndcg20
            # store best loss and save a model checkpoint
            torch.save(model.state_dict(), str(args.save_path) \
                       + '/epoch{}_embed{}_seq_len{}_{}_{}_{}.ckpt'.format(args.epoch, args.embed_dim, args.max_seq_len,
                                                                           epoch + 1, args.dataset_path.split('/')[-2],
                                                                           args.model))


def test(args, test_loader, model, repeat_matrix):
    model.eval()
    hr20s = []

    mrr20s = []
    ndcg20s = []

    hr10s = []
    mrr10s = []
    ndcg10s = []

    hr5s = []
    mrr5s = []
    ndcg5s = []
    repeat_time = []

    with torch.no_grad():
        for i, (uid, tar, neg_items, cur_s, last_s, neg_s, cur_ti, repeat_ti, neg_ti, cur_ig,
                repeat_ig, neg_ig, _) in tqdm(enumerate(test_loader), total=len(test_loader)):

            tar_lis = tar.numpy().tolist()
            cur_s_lis = cur_s.numpy().tolist()
            rt = [sum([tar_lis[i] == cur_s_lis[i][j] for j in range(len(cur_s_lis[i]))]) for i in range(len(tar_lis))]
            neg_items = neg_items.reshape(-1)
            cur_batch_size = cur_s.size(0)
            zero_matrix = [[0] * args.repeat_period_matrix_col] * args.repeat_period_matrix_row
            repeat_matrix_batch = torch.from_numpy(np.array([repeat_matrix[i] if i in repeat_matrix
                                                             else zero_matrix for i in tar.numpy()]))
            repeat_matrix_batch_neg = torch.from_numpy(np.array([repeat_matrix[i] if i in repeat_matrix else zero_matrix
                                                                 for i in neg_items.numpy()]))
            uid = uid.to(dtype=torch.long, device=device)
            tar = tar.to(dtype=torch.long, device=device)
            neg_items = neg_items.to(dtype=torch.long, device=device)
            repeat_matrix_batch = repeat_matrix_batch.to(dtype=torch.long, device=device)
            repeat_matrix_batch_neg = repeat_matrix_batch_neg.to(dtype=torch.long, device=device)

            cur_s, last_s, neg_s = (cur_s.to(dtype=torch.long, device=device),
                                    last_s.to(dtype=torch.long, device=device),
                                    neg_s.reshape(neg_s.size(0) * neg_s.size(1),
                                                  neg_s.size(2)).to(dtype=torch.long, device=device))

            cur_ti, repeat_ti, neg_ti = (cur_ti.to(dtype=torch.long, device=device),
                                         repeat_ti.to(dtype=torch.long, device=device),
                                         neg_ti.reshape(neg_ti.size(0) * neg_ti.size(1),
                                                        neg_ti.size(2)).to(dtype=torch.long, device=device))
            cur_ig, repeat_ig, neg_ig = (cur_ig.to(dtype=torch.long, device=device),
                                         repeat_ig.to(dtype=torch.long, device=device),
                                         neg_ig.reshape(neg_ig.size(0) * neg_ig.size(1),
                                                        neg_ig.size(2)).to(dtype=torch.long, device=device))
            neg_uid = uid.tile((args.negative_sample_test_num,)).reshape(args.negative_sample_test_num * uid.size(0),
                                                                         -1)
            neg_cur_s = cur_s.tile((args.negative_sample_test_num,)).reshape(
                args.negative_sample_test_num * cur_s.size(0), -1)
            neg_cur_ti = cur_ti.tile((args.negative_sample_test_num,)).reshape(
                args.negative_sample_test_num * cur_ti.size(0), -1)
            neg_cur_ig = cur_ig.tile((args.negative_sample_test_num,)).reshape(
                args.negative_sample_test_num * cur_ig.size(0), -1)
            outputs = ''
            if args.model == 'TSRec':
                tar_prediction = model(uid, tar, last_s, cur_s, repeat_ti, cur_ti, repeat_ig, cur_ig,
                                       repeat_matrix_batch)
                neg_prediction = model(neg_uid, neg_items, neg_s, neg_cur_s, neg_ti, neg_cur_ti, neg_ig, neg_cur_ig,
                                       repeat_matrix_batch_neg)
                tar_prediction = tar_prediction.unsqueeze(-1)
                neg_prediction = neg_prediction.reshape(cur_batch_size, -1)
            else:
                print('unknown model!')

            logits = torch.cat([tar_prediction, neg_prediction], dim=-1)
            hr20, mrr20, ndcg20 = metric.evaluate(logits, k=20)
            hr10, mrr10, ndcg10 = metric.evaluate(logits, k=10)
            hr5, mrr5, ndcg5 = metric.evaluate(logits, k=5)
            hr20s.append(hr20)
            hr10s.append(hr10)
            hr5s.append(hr5)
            mrr20s.append(mrr20)
            mrr10s.append(mrr10)
            mrr5s.append(mrr5)
            ndcg20s.append(ndcg20)
            ndcg10s.append(ndcg10)
            ndcg5s.append(ndcg5)
            repeat_time += rt

        mean_hr20 = np.mean(hr20s)
        mean_mrr20 = np.mean(mrr20s)
        mean_ndcg20 = np.mean(ndcg20s)

        mean_hr10 = np.mean(hr10s)
        mean_mrr10 = np.mean(mrr10s)
        mean_ndcg10 = np.mean(ndcg10s)

        mean_hr5 = np.mean(hr5s)
        mean_mrr5 = np.mean(mrr5s)
        mean_ndcg5 = np.mean(ndcg5s)
    return mean_hr20, mean_mrr20, mean_ndcg20, mean_hr10, mean_mrr10, mean_ndcg10, \
        mean_hr5, mean_mrr5, mean_ndcg5


if __name__ == '__main__':
    main()
