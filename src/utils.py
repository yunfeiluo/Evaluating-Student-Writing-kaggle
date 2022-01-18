import pandas as pd
from transformers import *

def form_output(folder, pred, IDS, MODEL_NAME = "bert-base-cased", MAX_LEN=1024):
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
    ind_to_label = dict()
    for key in label_to_ind:
        ind_to_label[label_to_ind[key]] = key
    
    # construct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    res = {
        'id': list(),
        'class': list(),
        'predictionstring': list(), # 2D array
    }
    for i in range(len(IDS)):
        if i % 1000 == 0:
            print(i)
        # read txt file
        filename = '../input/feedback-prize-2021/{}/{}.txt'.format(folder, IDS[i])
        txt = open(filename, 'r').read()
        total_len = len(txt)
        words = txt.split()
        
        # tokenize
        tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                       truncation=True, return_offsets_mapping=True)
        token_ids = tokens['input_ids']
        offsets = tokens['offset_mapping']
        
        # extract segments
        segments = dict() # map: (start, end) -> label
        offset_ind = 0
        start_ind = 0
        curr_label = ind_to_label[pred[i, 0]].split('_')[0]
        while offset_ind < len(token_ids) and token_ids[offset_ind] != 0:
            # get current token index
            t_start = offsets[offset_ind][0]
            t_end = offsets[offset_ind][1]
            
            label = ind_to_label[pred[i, offset_ind]].split('_')
            label_name = label[0]
            if len(label) > 1:
                pos = label[1]
            if label_name != curr_label or offset_ind == len(token_ids) - 1:
                segments[(start_ind, t_start)] = curr_label
                start_ind = t_start
                curr_label = label_name
            offset_ind += 1
        
        # form output
        curr_ind = 0
        for seg in segments:
            res['id'].append(IDS[i])
            res['class'].append(segments[seg])
            
            words = txt[seg[0]:seg[1]].split()
            pred_str = ' '.join([str(i + curr_ind) for i in range(len(words))])
            res['predictionstring'].append(pred_str)
            
            curr_ind += len(words)
        
    return pd.DataFrame(res)

def post_processing(folder, res_df, IDS):
    res = {
        'id': list(),
        'class': list(),
        'predictionstring': list(), # 2D array
    }
    
    def find_mode_label(arr):
        label_ct = dict()
        for label in arr:
            if label_ct.get(label) == None:
                label_ct[label] = 1
            else:
                label_ct[label] += 1
        label = sorted(label_ct.items(), key=lambda x: x[1])[-1][0]
        return label
    
    for i in range(len(IDS)):
#         if i % 1 == 0:
#             print(i)
        
        # construct labels array
        labels = list()
        curr_df = res_df.loc[res_df.id == IDS[i]]
        for index,row in curr_df.iterrows():
            pred_str = row.predictionstring.split()
            if len(pred_str) < 1:
                continue
            
            label = row['class']
            for s in pred_str:
                labels.append(label)
        
        # read txt file
        filename = '../input/feedback-prize-2021/{}/{}.txt'.format(folder, IDS[i])
        txt = open(filename, 'r').read().split()
        
        ending = ['.', '!', '?']
        sent_start_ind = 0
        curr_sentence = list()
        for j in range(len(txt)):
            if j >= len(labels):
                break
            
            word = txt[j]
            if word[-1] not in ending:
                curr_sentence.append(labels[j])
            else:
                curr_sentence.append(labels[j])
                label = find_mode_label(curr_sentence)
                
                if len(res['id']) > 0 and IDS[i] == res['id'][-1] and label == res['class'][-1]:
                    res['predictionstring'][-1] += ' ' + ' '.join([str(k+sent_start_ind) for k in range(len(curr_sentence))])
                else:
                    res['id'].append(IDS[i])
                    res['class'].append(label)
                    res['predictionstring'].append(' '.join([str(k+sent_start_ind) for k in range(len(curr_sentence))]))
                
                sent_start_ind += len(curr_sentence)
                curr_sentence = list()
        if len(curr_sentence) > 0:
            label = find_mode_label(curr_sentence)
            
            if len(res['id']) > 0 and label == res['class'][-1]:
                res['predictionstring'][-1] += ' ' + ' '.join([str(k+sent_start_ind) for k in range(len(curr_sentence))])
            else:
                res['id'].append(res['id'][-1])
                res['class'].append(label)
                res['predictionstring'].append(' '.join([str(k+sent_start_ind) for k in range(len(curr_sentence))]))
    return pd.DataFrame(res)