import pickle

from data_loader import *
from models import *


if __name__ == '__main__':
    with open('tokenized_data_bert-base-cased.pkl', 'rb') as f:
        saved = pickle.load(f)
        ids = saved['train_ids']
        attention = saved['train_attention']
        labels = saved['train_labels']

    np.random.seed(42)
    inds = [i for i in range(train_labels.shape[0])]
    np.random.shuffle(inds)

    split_point = int(len(inds) * 0.8)
    train_inds = inds[:split_point]
    val_inds = inds[split_point:]

    train_ids = ids[train_inds]
    train_attention = attention[train_inds]
    train_labels = labels[train_inds]

    val_ids = ids[val_inds]
    val_attention = attention[val_inds]
    val_labels = labels[val_inds]

    print('start training...')
    model = BERT_MLP()
    history = model.fit(
        {
            'input_ids': train_ids,
            'attention_mask': train_attention
        },
        labels=train_labels,
        epochs=2,
        batch_size=32,
        validation_data = (
            {
                'input_ids': val_ids,
                'attention_mask': val_attention
            },
            val_labels),
    )
    