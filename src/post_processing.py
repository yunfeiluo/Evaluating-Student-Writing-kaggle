import pandas as pd
from transformers import *

def word_to_label(folder, pred, IDS, MODEL_NAME = "bert-base-cased", MAX_LEN=1024):
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
    
    word_classes = {
        'id': list(),
        'classes': list()
    }
    for i in range(len(IDS)):
        word_classes['id'].append(IDS[i])
        
        # verbose
        if i % 1000 == 0:
            print(i)
            
        # read txt file
        filename = '../input/feedback-prize-2021/{}/{}.txt'.format(folder, IDS[i])
        txt = open(filename, 'r').read()
        txt_len = len(txt)
        words_num = len(txt.split())
        
        # tokenize
        tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                       truncation=True, return_offsets_mapping=True)
        token_ids = tokens['input_ids']
        offsets = tokens['offset_mapping']
        
        # extract word class
        word_class = list() # 1D array
        curr_words = list() # list of tuple (token_len, label_name)
        for j in range(MAX_LEN):
            # exit condition
            if len(word_class) == words_num:
                break
            
            # get current token index
            t_start = offsets[j][0]
            t_end = offsets[j][1]
            
            token_len = t_end - t_start
            curr_label = ind_to_label[pred[i, j]]
            curr_words.append((token_len, curr_label))
            
            # update word map                
            if t_end >= txt_len or txt[t_end] == ' ': # means the ending of a word
                if len(curr_words) < 2: # the word is not splitted
                    word_class.append(curr_label)
                else:
                    word_class.append(sorted(curr_words, key=lambda x: x[0])[-1][1])
                curr_words = list()
        word_classes['classes'].append(word_class)
        
    return word_classes

def form_raw_df(word_classes):
    res = {
        'id': list(),
        'class': list(),
        'predictionstring': list(), # 2D array
    }
    
    for i in range(len(word_classes['id'])):
        curr_seg = list() # (class, idx)
        for j in range(len(word_classes['classes'][i])):
            curr_class = word_classes['classes'][i][j].split('_')
            class_name = curr_class[0]
            pos = curr_class[1] if len(curr_class) > 1 else 'i'
            
#             if len(curr_seg) < 1 or class_name == curr_seg[-1][0]:
#                 curr_seg.append((class_name, j))
#             else:
#                 res['id'].append(word_classes['id'][i])
#                 res['class'].append(curr_seg[-1][0])
#                 res['predictionstring'].append(' '.join([str(k[1]) for k in curr_seg]))
#                 curr_seg = [(class_name, j)]

            if pos == 'b': # if it's the begining of a segment
                if len(curr_seg) < 1: # haven't init
                    curr_seg.append((class_name, j))
                else: # the close of previous segment
                    res['id'].append(word_classes['id'][i])
                    res['class'].append(curr_seg[-1][0])
                    res['predictionstring'].append(' '.join([str(k[1]) for k in curr_seg]))
                    curr_seg = [(class_name, j)]
            elif len(curr_seg) >= 1 and class_name == curr_seg[-1][0]: # if it's the inside of a segment with same class as begining
                curr_seg.append((class_name, j))
            elif len(curr_seg) >= 1:
                res['id'].append(word_classes['id'][i])
                res['class'].append(curr_seg[-1][0])
                res['predictionstring'].append(' '.join([str(k[1]) for k in curr_seg]))
                curr_seg = list()
            else:
                curr_seg = list()
    
    return pd.DataFrame(res)

def post_processing_mode(folder, word_classes): # determine the class of a sentence by mode of label
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
    
    for i in range(len(word_classes['id'])):
        # read txt file
        filename = '../input/feedback-prize-2021/{}/{}.txt'.format(folder, word_classes['id'][i])
        words = open(filename, 'r').read().split()
        
        ending = ['.', '!', '?']
        curr_sentence = list() # list of tuples: (word_idx, class_name)
        for j in range(len(word_classes['classes'][i])):
            curr_sentence.append((j, word_classes['classes'][i][j]))
            word = words[j]
            if word[-1] in ending:
                label = find_mode_label([k[1].split('_')[0] for k in curr_sentence])
                
                if len(res['id']) > 0 and word_classes['id'][i] == res['id'][-1] and label == res['class'][-1]:
                    res['predictionstring'][-1] += ' ' + ' '.join([str(k[0]) for k in curr_sentence])
                else:
                    res['id'].append(word_classes['id'][i])
                    res['class'].append(label)
                    res['predictionstring'].append(' '.join([str(k[0]) for k in curr_sentence]))
                
                # clear up
                curr_sentence = list()
                
        if len(curr_sentence) > 0:
            label = find_mode_label([k[1].split('_')[0] for k in curr_sentence])
            
            if len(res['id']) > 0 and word_classes['id'][i] == res['id'][-1] and label == res['class'][-1]:
                res['predictionstring'][-1] += ' ' + ' '.join([str(k[0]) for k in curr_sentence])
            else:
                res['id'].append(word_classes['id'][i])
                res['class'].append(label)
                res['predictionstring'].append(' '.join([str(k[0]) for k in curr_sentence]))
    return pd.DataFrame(res)
    