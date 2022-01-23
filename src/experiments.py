import pickle

from utils import *
from models import *
from post_processing import *


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # USE MULTIPLE GPUS
    if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
        strategy = tf.distribute.get_strategy()
        print('single strategy')
    else:
        strategy = tf.distribute.MirroredStrategy()
        print('multiple strategy')

    # ============================== SPLIT_LINE ==================================

    # MODEL_NAME = "bert-base-cased"
    # MODEL_NAME = "../input/feedbacksaved/BERT" # load from pretrained.
    # MAX_LEN = 512

    MODEL_NAME = 'allenai/longformer-base-4096'
    # MODEL_NAME = '../input/feedbacksaved/LongFormer'
    MAX_LEN = 1024
    LR=0.25e-4

    # # processing data
    # train_ids, train_attention, train_labels = load_train_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)

    # with open('tokenized_data_longformer.pkl', 'wb') as f:
    #     saved = {
    #         'train_ids': train_ids,
    #         'train_attention': train_attention,
    #         'train_labels': train_labels
    #     }
    #     pickle.dump(saved, f)

    # load saved data and build model
    with open('../input/feedbacksaved/tokenized_data_longformer.pkl', 'rb') as f:
        saved = pickle.load(f)
        ids = saved['train_ids'][:, :MAX_LEN]
        attention = saved['train_attention'][:, :MAX_LEN]
        labels = saved['train_labels'][:, :MAX_LEN, :]

    print('input seq shape', ids.shape)
    print('attention shape', attention.shape)
    print('labels shape', labels.shape)

    with strategy.scope():
        model = build_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, LR=LR)
    
    # ============================== SPLIT_LINE ==================================
    
    # # load trained model if available
    # model.load_weights('../input/feedbacksaved/models/bert_mlp.h5')

    # # train-val model
    # train_val(model, ids, attention, labels, 
    #           train_size=0.8, 
    #           epochs=5,
    #           batch_size=16,
    #           saved_name='saved_model.h5')

    # train on entire training set
    train_val(model, ids, attention, labels, 
            train_size=1.0, 
            epochs=5,
            batch_size=4,
            saved_name='{}_entire.h5'.format(MODEL_NAME.split('/')[-1]))
    
    # ============================== SPLIT_LINE ==================================

    # MODEL_NAME = "bert-base-cased"
    # MODEL_NAME = "../input/feedbacksaved/BERT" # load from pretrained.
    # MAX_LEN = 512

    # MODEL_NAME = 'allenai/longformer-base-4096'
    MODEL_NAME = '../input/feedbacksaved/LongFormer'
    MAX_LEN = 1024

    # build and load model
    with strategy.scope():
        model = build_model(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
    model.load_weights('../input/feedbacksaved/models/longformer_mlp.h5')
    print('Model Loading Complete.')

    # load test data
    test_ids, test_attention, test_IDS = load_test_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
    print('Test Data Loading Complete.')

    # make prediction
    test_pred = model.predict([test_ids, test_attention], batch_size=16, verbose=2).argmax(axis=-1)
    print('Prediction Complete.')

    test_res_int = get_preds(dataset='test', verbose=False, text_ids=test_IDS, preds=test_pred)

    # write to file
    test_res_int.to_csv('submission.csv',index=False)