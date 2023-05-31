import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import math
import random

def load_data(root_path, seq_len):
    '''Loads the dataset

    type path: String


    '''

    # Load the dataset
    path_train_data = root_path + 'train_data_seq_'+str(seq_len)
    path_test_data = root_path + 'test_data_seq_'+str(seq_len)
    path_min_time_interval = root_path + 'repeat_min_interval'
    with open(path_train_data, 'rb') as f1:
        train_set = pickle.load(f1)

    with open(path_test_data, 'rb') as f2:
        test_set = pickle.load(f2)

    with open(path_min_time_interval, 'rb') as f3:
        repeat_min_interval = pickle.load(f3)
    return train_set, test_set, repeat_min_interval


def load_matrix(root_path):
    '''Loads the repeat matrix

    type path: String


    '''

    path_repeat_matrix = root_path + 'repeat_matrix'
    with open(path_repeat_matrix, 'rb') as f1:
        repeat_matrix = pickle.load(f1)
    return repeat_matrix


class RecDataset(Dataset):
    """define the pytorch Dataset class.
    """

    def __init__(self, data, repeat_min_interval, max_time_inter, max_len, use_seq=False):
        self.data = data
        self.repeat_min_interval = repeat_min_interval
        self.max_time_inter = max_time_inter

        self.mask_prob = 0.8
        self.max_len = max_len
        self.use_seq = use_seq
        print('-' * 50)
        print('Dataset info:')
        print('Number of seqs: {}'.format(len(data)))
        print('-' * 50)

    def __getitem__(self, index):
        user_id = np.array(self.data[index][0]['uid'])
        target_item = np.array(self.data[index][0]['target_item'])
        negative_items = np.array(self.data[index][0]['negative_items'])
        cur_item_seq = np.array(self.data[index][0]['cur_item_seq'])
        last_item_seq = np.array(self.data[index][0]['last_item_seq'])
        last_negative_item_seq = np.array(self.data[index][0]['last_negative_item_seq'])

        if self.use_seq:
            len = (cur_item_seq!=0).sum()
            target_items_seq = np.concatenate ([cur_item_seq[1:len], [target_item]], axis=0)
            pad= np.zeros((self.max_len-target_items_seq.shape[0]))
            target_items_seq = np.concatenate([target_items_seq, pad], axis=0)
            neg_items_seq = np.repeat([negative_items], self.max_len,axis=1).reshape(-1, self.max_len)


        cur_time_interval = np.array(self.data[index][0]['cur_time_interval'])
        repeat_time_interval = np.array(self.data[index][0]['repeat_time_interval'])
        negative_repeat_time_interval = np.array(self.data[index][0]['negative_repeat_time_interval'])
        cur_index_gap = np.array(self.data[index][0]['cur_index_gap'])
        repeat_index_gap = np.array(self.data[index][0]['repeat_index_gap'])
        negative_repeat_index_gap = np.array(self.data[index][0]['negative_repeat_index_gap'])
        if self.data[index][0]['target_item'] in self.repeat_min_interval:
            min_time = self.repeat_min_interval[self.data[index][0]['target_item']]
        else:
            min_time = cur_time_interval.item()+1e-9

        cur_time_interval = math.ceil(cur_time_interval / min_time)
        cur_time_interval = np.array([min(self.max_time_inter, cur_time_interval)])
        repeat_time_interval_norm = np.where(np.ceil(repeat_time_interval / min_time) > self.max_time_inter - 1,
                                        self.max_time_inter - 1, np.ceil(repeat_time_interval / min_time))
        negative_repeat_time_interval = np.where(
            np.ceil(negative_repeat_time_interval / min_time) > self.max_time_inter - 1,
            self.max_time_inter - 1, np.ceil(negative_repeat_time_interval / min_time))
        if self.use_seq:
            return user_id, target_item, negative_items, cur_item_seq, last_item_seq, last_negative_item_seq, \
                   cur_time_interval, repeat_time_interval_norm, negative_repeat_time_interval, cur_index_gap, \
                   repeat_index_gap, negative_repeat_index_gap, target_items_seq, neg_items_seq
        else:
            return user_id, target_item, negative_items, cur_item_seq, last_item_seq, last_negative_item_seq, \
                   cur_time_interval, repeat_time_interval_norm, negative_repeat_time_interval, cur_index_gap, \
                    repeat_index_gap, negative_repeat_index_gap, repeat_time_interval

    def __len__(self):
        return len(self.data)
