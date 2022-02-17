import os
import pickle

import numpy as np
import pandas as pd
from transformers import *

from torch.utils.data import DataLoader, Dataset

class FeedbackDataset(Dataset):
    def __init__(self, samples, mask, labels):
        self.samples = samples
        self.masks = mask
        self.labels = labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_pack = {
            'input_ids': self.samples[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx],
        }

        return data_pack

def load_train_data(val_size=0, MODEL_NAME="allenai/longformer-base-4096", MAX_LEN=1024):
    # construct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # load csv file
    df = pd.read_csv('../input/feedback-prize-2021/train.csv')
    IDS = df.id.unique()
    train_ids = np.zeros((len(IDS), MAX_LEN), dtype='int32')
    train_attention = np.zeros((len(IDS), MAX_LEN), dtype='int32')
    
    # init labels
    label_to_ind = {
        'Lead_b': 0,
        'Lead_i': 1,
        'Position_b': 2,
        'Position_i': 3,
        'Evidence_b': 4,
        'Evidence_i': 5,
        'Claim_b': 6,
        'Claim_i': 7,
        'Concluding Statement_b': 8,
        'Concluding Statement_i': 9,
        'Counterclaim_b': 10,
        'Counterclaim_i': 11,
        'Rebuttal_b': 12,
        'Rebuttal_i': 13,
        'other': 14
    }    
    train_labels = np.zeros((len(IDS), MAX_LEN, len(label_to_ind)), dtype='int32')
    
    # form samples
    for i in range(len(IDS)):
        if i % 1000 == 0:
            print(i)
        # read txt file
        filename = '../input/feedback-prize-2021/train/{}.txt'.format(IDS[i])
        txt = open(filename, 'r').read()
        words = txt.split()
        
        # tokenize
        tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                       truncation=True, return_offsets_mapping=True)
        train_ids[i, :] = tokens['input_ids']
        train_attention[i, :] = tokens['attention_mask']
        offsets = tokens['offset_mapping']
        
        # extract labels for each token
        curr_df = df.loc[df.id==IDS[i]]
        offset_ind = 0
        for index,row in curr_df.iterrows():
            label = row.discourse_type + '_b'
            
            w_start = row.discourse_start
            w_end = row.discourse_end
            
            if offset_ind >= len(offsets):
                break
            
            # set labels
            t_start = offsets[offset_ind][0]
            while w_end > t_start:
                # exit condition
                if offset_ind >= len(offsets):
                    break
                
                # get current token index
                t_start = offsets[offset_ind][0]
                t_end = offsets[offset_ind][1]
                
                # set label if within range
                if t_end <= w_end:
                    train_labels[i, offset_ind, label_to_ind[label]] = 1
                    label = row.discourse_type + '_i'
                
                # update global var(s)
                offset_ind += 1
    train_labels[:, :, 14] = 1 - np.max(train_labels, axis=-1)

    # construct dataset object
    if val_size == 0:
        train_dataset = FeedbackDataset(samples=train_ids, mask=train_attention, labels=train_labels)
        return train_dataset, None
    
    inds = [i for i in range(len(train_ids))]
    np.random.seed(42)
    np.random.shuffle(inds)
    split_ind = int(len(inds) * val_size)
    train_inds = inds[split_ind:]
    val_inds = inds[:split_ind]

    train_dataset = FeedbackDataset(samples=train_ids[train_inds], mask=train_attention[train_inds], labels=train_labels[train_inds])
    val_dataset = FeedbackDataset(samples=train_ids[val_inds], mask=train_attention[val_inds], labels=train_labels[val_inds])
    
    return train_dataset, val_dataset

def load_test_data(MODEL_NAME="allenai/longformer-base-4096", MAX_LEN=1024, batch_size=4):
    # construct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    IDS = os.listdir('../input/feedback-prize-2021/test')
    IDS = [i.split('.')[0] for i in IDS]
    test_ids = np.zeros((len(IDS), MAX_LEN), dtype='int32')
    test_attention = np.zeros((len(IDS), MAX_LEN), dtype='int32')
    
    # form samples
    for i in range(len(IDS)):
        if i % 1000 == 0:
            print(i)
        # read txt file
        filename = '../input/feedback-prize-2021/test/{}.txt'.format(IDS[i])
        txt = open(filename, 'r').read()
        
        # tokenize
        tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                       truncation=True, return_offsets_mapping=True)
        test_ids[i, :] = tokens['input_ids']
        test_attention[i, :] = tokens['attention_mask']
    
    test_dataset = FeedbackDataset(samples=train_ids, mask=train_attention, labels=train_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # change shuflle here if do not wanna shuffle
    return test_dataloader, IDS

if __name__ == '__main__':
    # config
    MODEL_NAME = "allenai/longformer-base-4096"
    MAX_LEN = 1024

    # load data
    train_ids, train_attention, train_labels = load_train_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)

    # save data
    with open('saved/tokenized_data_{}_{}.pkl'.format(MODEL_NAME, MAX_LEN), 'wb') as f:
        saved = {
            'train_ids': train_ids,
            'train_attention': train_attention,
            'train_labels': train_labels
        }
        pickle.dump(saved, f)