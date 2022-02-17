from transformers import *
import tensorflow as tf

# connection port
def build_model(MODEL_NAME="allenai/longformer-base-4096", MAX_LEN=1024, LR=1e-4):
    model = LongFormer(MODEL_NAME=MODEL_NAME, MAX_LEN=MAX_LEN, LR=LR) # baseline
    return model

# define models
def LongFormer(MODEL_NAME="allenai/longformer-base-4096", MAX_LEN=1024, LR=1e-4):
    # construct input
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')
    
    # pretrained/finetuned model (Transformers)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    backbone = TFAutoModel.from_pretrained(MODEL_NAME, config=config)
    backbone.trainable = True
    
#     # save the model
#     os.mkdir('model')
#     backbone.save_pretrained('model')
#     config.save_pretrained('model')
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.save_pretrained('model')
    
    # downstream output layer(s)
    out = backbone(input_ids, attention_mask=mask)
    out = tf.keras.layers.Dense(256, activation='relu')(out[0])
    out1 = tf.keras.layers.Dense(15, activation='softmax', dtype='float32')(out)
    out2 = tf.keras.layers.Dense(15, activation='softmax', dtype='float32')(out)
    output = [out1, out2]
    
    # define loss
    loss1 = lambda x, y: 1.0 * tf.keras.losses.CategoricalCrossentropy()(x, y)
    loss2 = lambda x, y: 1.0 * tf.keras.losses.CategoricalCrossentropy()(x, y)
    loss = [loss1, loss2]
    
    # integration
    model = tf.keras.Model(inputs=[input_ids,mask], outputs=output)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LR),
                  loss = loss,
                  metrics = [tf.keras.metrics.CategoricalAccuracy()])
    
    return model
