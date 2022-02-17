import pickle

from utils import *
from models import *
from post_processing import *

if __name__ == '__main__':
    # MODEL_NAME = 'allenai/longformer-base-4096'
    MODEL_NAME = '../input/feedbacksaved/LongFormer'
    MAX_LEN = 1024

    # build and load model
    model = build_model()
    model.load_weights('../input/feedbacksaved/models/longformer_mlp.h5')
    print('Model Loading Complete.')

    # load test data
    test_ids, test_attention, test_IDS = load_test_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
    print('Test Data Loading Complete.')

    # make prediction
    test_pred = model.predict([test_ids, test_attention], batch_size=BATCH_SIZE, verbose=2)[0].argmax(axis=-1)
    print('Prediction Complete.')

    test_res_int = get_preds(dataset='test', verbose=False, text_ids=test_IDS, preds=test_pred)

    # write to file
    test_res_int.to_csv('submission.csv',index=False)