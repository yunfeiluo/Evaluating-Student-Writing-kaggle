import os
import pickle
import tqdm

import numpy as np
import pandas as pd

def train_val(
    model, 
    optimizer,
    train_dataloader,
    val_dataloader,
    device,
    epochs=5,
    verbose=True
    ):

    best_score = np.inf
    train_loss = list()
    print('Training start..')
    for e in tqdm.tqdm(range(epochs)):
        # training
        model.train()
        for data_pack in train_dataloader:
            # move to device
            for key in data_pack:
                data_pack[key] = data_pack[key].to(device)
            
            # forward
            loss = model(data_pack)

            # backprop
            # print('backprop...')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # updata var
            train_loss.append(loss.cpu().detach().item())
        
        # verbose
        if verbose:
            print('Train Epoch {}, last loss {}, best loss {}.'.format(e, train_loss[-1], np.min(train_loss)))
        
        # validation
        model.eval()
        val_pred = list()
        val_true = list()
        for data_pack in val_dataloader:
            # move to device
            for key in data_pack:
                data_pack[key] = data_pack[key].to(device)

            # foward
            y_pred, y_true = model.predict(data_pack)

             # uypdate var
            val_true += y_true.cpu().detach().numpy().tolist()
            val_pred += y_pred.cpu().detach().numpy().tolist()
        
        # calculate evaluation metric
        score = (np.array(val_pred) == np.array(val_true)).mean()
        
         # update global vars
        if score < best_score:
            best_score = score
        
        # verbose
        if verbose:
            print('Epoch {}, Val eval: {}, Best eval: {}'.format(e, score, best_score))

if __name__ == '__main__':
    print('Check')