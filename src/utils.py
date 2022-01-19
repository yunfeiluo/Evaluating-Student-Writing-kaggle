import os
import pickle

import numpy as np
import pandas as pd
from transformers import *

# functions for loading and train/val data

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

def load_test_data(MODEL_NAME="bert-base-cased", MAX_LEN=1024):
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
    
    return test_ids, test_attention, IDS

def train_val(model, ids, attention, labels, 
              train_size=0.8, 
              epochs=5,
              batch_size=32,
              saved_name='saved_model.h5'
             ):
    # TRAIN VALID SPLIT 80% 20%
    np.random.seed(42)
    IDS = pd.read_csv('../input/feedback-prize-2021/train.csv').id.unique()
    inds = [i for i in range(len(IDS))]
    np.random.shuffle(inds)
    split_point = int(train_size * len(inds))
    train_idx = inds[:split_point]
    val_idx = inds[split_point:]
    print('Train size',len(train_idx),', Valid size',len(val_idx))

    print('start training...')
    model.fit(x = [ids[train_idx,], attention[train_idx,]],
              y = labels[train_idx,],
              validation_data = ([ids[val_idx,], attention[val_idx,]],
                                 labels[val_idx,]),
              epochs = epochs,
              batch_size = batch_size,
              verbose = 2)

    # SAVE MODEL WEIGHTS
    model.save_weights(saved_name)

if __name__ == '__main__':
    # # config
    # MODEL_NAME = "bert-base-cased"
    # MODEL_NAME = "../input/feedbacksaved/BERT" # load from pretrained.
    # MAX_LEN = 512

    MODEL_NAME = 'allenai/longformer-base-4096'
    # MODEL_NAME = '../input/feedbacksaved/LongFormer'
    MAX_LEN = 1024

    # processing data
    train_ids, train_attention, train_labels = load_train_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)

    with open('tokenized_data_longformer.pkl', 'wb') as f:
        saved = {
            'train_ids': train_ids,
            'train_attention': train_attention,
            'train_labels': train_labels
        }
        pickle.dump(saved, f)