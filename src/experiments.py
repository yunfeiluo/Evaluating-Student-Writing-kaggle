import pickle

from utils import *
from models import *
from post_processing import *


if __name__ == '__main__':
    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)

    # ============================== SPLIT_LINE ==================================

    MODEL_NAME = 'allenai/longformer-base-4096'
    # MODEL_NAME = '../input/feedbacksaved/LongFormer'
    MAX_LEN = 1024

    LR=0.25e-4
    BATCH_SIZE = 4
    EPOCHS = 5

    # processing data
    train_dataloader, val_dataloader = load_train_data(
        batch_size=BATCH_SIZE,
        val_size=0.2,
        MODEL_NAME=MODEL_NAME, 
        MAX_LEN=MAX_LEN,
        )

    with open('train_val_loader_longformer.pkl', 'wb') as f:
        saved = {
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
        }
        pickle.dump(saved, f)

    # # load saved data and build model
    # with open('train_val_loader_longformer.pkl', 'rb') as f:
    #     saved = pickle.load(f)
    #     train_dataloader = saved['train_dataloader']
    #     val_dataloader = saved['val_dataloader']

    # construct model
    model = build_model()
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    #     momentum=0.9, 
    #     nesterov=True
    )
    
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
    train_val(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        device,
        epochs=EPOCHS,
        verbose=True
    )
    
    # ============================== SPLIT_LINE ==================================

    # # MODEL_NAME = 'allenai/longformer-base-4096'
    # MODEL_NAME = '../input/feedbacksaved/LongFormer'
    # MAX_LEN = 1024

    # # build and load model
    # model = build_model()
    # model.load_weights('../input/feedbacksaved/models/longformer_mlp.h5')
    # print('Model Loading Complete.')

    # # load test data
    # test_ids, test_attention, test_IDS = load_test_data(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN)
    # print('Test Data Loading Complete.')

    # # make prediction
    # test_pred = model.predict([test_ids, test_attention], batch_size=16, verbose=2).argmax(axis=-1)
    # print('Prediction Complete.')

    # test_res_int = get_preds(dataset='test', verbose=False, text_ids=test_IDS, preds=test_pred)

    # # write to file
    # test_res_int.to_csv('submission.csv',index=False)