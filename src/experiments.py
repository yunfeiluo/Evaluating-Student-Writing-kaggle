import pickle

from data_loader import *
from models import *


if __name__ == '__main__':
    # config
    MODEL_NAME = "/bert-base-cased"
    MAX_LEN = 512
    
    # load data
    with open('saved/tokenized_data_bert-base-cased.pkl', 'rb') as f:
        saved = pickle.load(f)
        ids = saved['train_ids'][:, :MAX_LEN]
        attention = saved['train_attention'][:, :MAX_LEN]
        labels = saved['train_labels'][:, :MAX_LEN, :]

    print('input seq shape', ids.shape)
    print('attention shape', attention.shape)
    print('labels shape', labels.shape)

    # construct model
    model = BERT_MLP(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
    model.summary()
    
    # TRAIN VALID SPLIT 80% 20%
    np.random.seed(42)
    IDS = pd.read_csv('data/train.csv').id.unique()
    inds = [i for i in range(len(IDS))]
    np.random.shuffle(inds)
    split_point = int(0.8 * len(inds))
    train_idx = inds[:split_point]
    val_idx = inds[split_point:]
    print('Train size',len(train_idx),', Valid size',len(val_idx))

    print('start training...')
    model.fit(x = [ids[train_idx,], attention[train_idx,]],
            y = labels[train_idx,],
            validation_data = ([ids[val_idx,], attention[val_idx,]],
                                labels[val_idx,]),
            epochs = 5,
            batch_size = 32,
            verbose = 2)

    # SAVE MODEL WEIGHTS
    model.save_weights('saved/{}_mlp_{}.h5'.format(MODEL_NAME, MAX_LEN))