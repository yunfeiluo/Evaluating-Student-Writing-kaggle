import pickle

import numpy as np
import pandas as pd
from transformers import *

def load_train_data(MODEL_NAME="bert-base-cased", MAX_LEN=1024):
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
    return train_ids, train_attention, train_labels

if __name__ == '__main__':
    # config
    MODEL_NAME = "/bert-base-cased"
    MAX_LEN = 512

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