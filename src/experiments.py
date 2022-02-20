import pickle

from utils import *
from models import *
from post_processing import *

if __name__ == '__main__':
    # use gpu if available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # USE MULTIPLE GPUS
    if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
        strategy = tf.distribute.get_strategy()
        print('single strategy')
    else:
        strategy = tf.distribute.MirroredStrategy()
        print('multiple strategy')

    # ============================== SPLIT_LINE ==================================

    MODEL_NAME = 'allenai/longformer-base-4096'
    # MODEL_NAME = '../input/feedbacksaved/LongFormer'
    MAX_LEN = 1024

    LR=0.25e-4
    BATCH_SIZE = 4
    EPOCHS = 5

    # processing data
    ids, attention, labels = load_train_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)

    with open('tokenized_data_longformer.pkl', 'wb') as f:
        saved = {
            'train_ids': train_ids,
            'train_attention': train_attention,
            'train_labels': train_labels
        }
        pickle.dump(saved, f)

    # # load saved data and build model
    # with open('../input/feedbacksaved/tokenized_data_longformer.pkl', 'rb') as f:
    #     saved = pickle.load(f)
    #     ids = saved['train_ids'][:, :MAX_LEN]
    #     attention = saved['train_attention'][:, :MAX_LEN]
    #     labels = saved['train_labels'][:, :MAX_LEN, :]

    print('input seq shape', ids.shape)
    print('attention shape', attention.shape)
    print('labels shape', labels.shape)

    # construct model
    with strategy.scope():
        model = build_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, LR=LR)
    print(model.summary())

    # construct labels
    coarse_labels = coarse_class(labels)
    binary_labels = binary_class(labels)
    
    # ============================== SPLIT_LINE ==================================

    # TRAIN VALID SPLIT 80% 20%
    train_size = 0.8

    # split dataset
    np.random.seed(42)
    inds = [i for i in range(len(ids))]
    np.random.shuffle(inds)
    split_point = int(train_size * len(inds))
    train_idx = inds[:split_point]
    val_idx = inds[split_point:]
    print('Train size',len(train_idx),', Valid size',len(val_idx))

    val_labels = [labels[val_idx,]]
    # val_labels = [labels[val_idx,], coarse_labels[val_idx,]]
    # val_labels = [labels[val_idx,], binary_labels[val_idx,]]
    # val_labels = [labels[val_idx,], coarse_labels[val_idx,], binary_labels[val_idx,]]

    print('start training...')
    model.fit(x = [ids[train_idx,], attention[train_idx,]],
              y = [labels[train_idx,], labels[train_idx,]], # custom labels
              validation_data = ([ids[val_idx,], attention[val_idx,]],
                                 val_labels),
              epochs = EPOCHS,
              batch_size = BATCH_SIZE,
              verbose = 2)

    # SAVE MODEL WEIGHTS
    model.save_weights('saved_model.h5')
    