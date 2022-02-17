import pickle

from utils import *
from models import *
from post_processing import *

from torch.utils.data import DataLoader

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
    train_dataset, val_dataset = load_train_data(
        val_size=0.2,
        MODEL_NAME=MODEL_NAME, 
        MAX_LEN=MAX_LEN,
        )

    with open('train_val_dataset_longformer.pkl', 'wb') as f:
        saved = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
        }
        pickle.dump(saved, f)
    
    # # load saved data and build model
    # with open('train_val_dataset_longformer.pkl', 'rb') as f:
    #     saved = pickle.load(f)
    #     train_dataset = saved['train_dataset']
    #     val_dataset = saved['val_dataset']
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_dataloader = list()

    # construct model
    model = build_model().to(device)
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # # load trained model if available
    # model.load_state_dict(torch.load(PATH))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    #     momentum=0.9, 
    #     nesterov=True
    )
    
    # ============================== SPLIT_LINE ==================================

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

    torch.save(model.state_dict(), 'saved_trained.model')
    
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