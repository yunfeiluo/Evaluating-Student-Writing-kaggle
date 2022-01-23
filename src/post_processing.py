import pandas as pd
from transformers import *

# =====================================================================
def get_preds(dataset='train', verbose=True, text_ids=None, preds=None):
    target_map_rev = {0: 'Lead', 1: 'Position', 2: 'Evidence', 3: 'Claim', 4: 'Concluding Statement', 5: 'Counterclaim', 6: 'Rebuttal', 7: 'blank'}
    
    # construct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    all_predictions = list()
    for id_num in range(len(preds)):
#         if (id_num % 100 == 0) & (verbose): print(id_num, ', ', end = '')

        # read and tokenize txt
        n = text_ids[id_num]
        name = f'../input/feedback-prize-2021/{dataset}/{n}.txt'
        txt = open(name, 'r').read()
        tokens = tokenizer.encode_plus(txt, max_length = MAX_LEN, padding = 'max_length', truncation = True, return_offsets_mapping = True)
        off = tokens['offset_mapping']
        
        # find the start of each word
        w = list()
        blank = True
        for i in range(len(txt)):
            if (txt[i] != ' ') & (txt[i] != '\n') & (blank == True):
                w.append(i)
                blank = False
            elif (txt[i] == ' ') | (txt[i] == '\n'):
                blank = True
        w.append(1e6)
        
        # create token_to_word map
        word_map = -1 * np.ones(MAX_LEN, dtype = 'int32')
        w_i = 0
        for i in range(len(off)):
            if off[i][1] == 0:
                continue
            while off[i][0] >= w[w_i + 1]: 
                w_i += 1
            word_map[i] = int(w_i)
        
        # retrieve the segments and class
        pred = preds[id_num,] / 2.0
        i = 0
        while i < MAX_LEN:
            prediction = list()
            start = pred[i]
            if start in [0, 1, 2, 3, 4, 5, 6, 7]:
                prediction.append(word_map[i])
                i += 1
                if i >= MAX_LEN: 
                    break
                while pred[i] == start + 0.5:
                    if not word_map[i] in prediction: 
                        prediction.append(word_map[i])
                    i += 1
                    if i >= MAX_LEN: 
                        break
            else: 
                i += 1
            prediction = [x for x in prediction if x != -1]
            if len(prediction) > 4: 
                all_predictions.append((n, target_map_rev[int(start)], ' '.join([str(x) for x in prediction])))

    # MAKE DATAFRAME
    df = pd.DataFrame(all_predictions)
    df.columns = ['id', 'class', 'predictionstring']
    return df
        