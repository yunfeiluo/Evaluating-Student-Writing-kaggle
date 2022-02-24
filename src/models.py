import os

import numpy as np
from transformers import *
import tensorflow as tf

def download_save_model(MODEL_NAME="allenai/longformer-base-4096"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    backbone = TFAutoModel.from_pretrained(MODEL_NAME, config=config)
    backbone.trainable = True
    
    # save the model
    os.mkdir('model')
    backbone.save_pretrained('model')
    config.save_pretrained('model')
    tokenizer.save_pretrained('model')

# connection port
def build_model(MODEL_NAME="allenai/longformer-base-4096", MAX_LEN=1024, LR=1e-4):
    # model = LongFormer(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, LR=LR) # baseline
    model = LongFormerMultitask(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, LR=LR) # baseline
    return model

# define models
def LongFormerMultitask(MODEL_NAME="allenai/longformer-base-4096", MAX_LEN=1024, LR=1e-4):
    # construct input
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')
    
    # pretrained/finetuned model (Transformers)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    backbone = TFAutoModel.from_pretrained(MODEL_NAME, config=config)
    backbone.trainable = True
    
    # downstream output layer(s)
    out = backbone(input_ids, attention_mask=mask)[0]

    # multitask configuration
#     tasks = ["main_task"]
#     out_size = [15]
#     tasks_weight = [1.0]
    
    tasks = ["main_task", "coarse_class", "binary_class"]
    out_size = [15, 7, 3]
    # tasks_weight = [1.0, 0.6, 0.4]
    tasks_weight = np.exp(out_size) / np.exp(out_size).sum()

    # construct multihead output
    outputs = list()
    loss = dict()
    loss_weights = dict()
    mets = dict()

    for i in range(len(tasks)):
        subout = tf.keras.layers.Dense(256, activation='relu')(out)
        outputs.append(tf.keras.layers.Dense(out_size[i], activation='softmax', dtype='float32', name=tasks[i])(subout))
        loss[tasks[i]] = tf.keras.losses.CategoricalCrossentropy()
        loss_weights[tasks[i]] = tasks_weight[i]
        mets[tasks[i]] = tf.keras.metrics.CategoricalAccuracy()
    
    # integration
    model = tf.keras.Model(inputs=[input_ids,mask], outputs=outputs)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LR),
                  loss = loss,
                  metrics = mets,
                  loss_weights = loss_weights
                 )
    
    return model

def coarse_class(labels):
    N, L, C = labels.shape

    # 0, 1 are Argument;, 2, 3 are Declaration; 4, 5 are Evidence; 6 is other
    ind_to_ind = {
        6: 0, 
        7: 1,
        10: 0, 
        11: 1, 
        12: 0, 
        13: 1,
        0: 2,
        1: 3,
        2: 2, 
        3: 3,
        8: 2, 
        9: 3, 
        4: 4, 
        5: 5,
        14: 6
    }
    coarse_labels = np.zeros((N, L, 7), dtype='int32')
    old_labels = labels.argmax(dim=2)
    for i in range(N):
        for j in range(L):
            coarse_labels[i, j][ind_to_ind[old_labels[i, j]]] = 1
    return coarse_labels

def binary_class(labels):
    N, L, C = labels.shape

    # 0 for begin, 1 for inside, 2 for other
    ind_to_ind = {14: 2}
    for i in range(14):
        ind_to_ind[i] = 0 if i % 2 == 0 else 1

    binary_labels = np.zeros((N, L, 3), dtype='int32')
    old_labels = labels.argmax(dim=2)
    for i in range(N):
        for j in range(L):
            binary_labels[i, j][ind_to_ind[old_labels[i, j]]] = 1
    return binary_labels
