import tensorflow as tf
from transformers import *

def BERT_MLP(MODEL_NAME="bert-base-cased", MAX_LEN=1024):
    # construct input
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')
    
    # pretrained model (Transformers)
    config = AutoConfig.from_pretrained(MODEL_NAME) 
    backbone = TFAutoModel.from_pretrained(MODEL_NAME, config=config)
    backbone.trainable = False
    
    # downstream output layer(s)
    x = backbone(input_ids, attention_mask=mask)
    x = tf.keras.layers.Dense(256, activation='relu')(x[0])
    x = tf.keras.layers.Dense(15, activation='softmax', dtype='float32')(x)
    
    # integration
    model = tf.keras.Model(inputs=[input_ids,mask], outputs=x)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
                  loss = [tf.keras.losses.CategoricalCrossentropy()],
                  metrics = [tf.keras.metrics.CategoricalAccuracy()])
    
    return model