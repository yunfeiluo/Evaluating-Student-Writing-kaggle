import torch
import torch.nn as nn
from transformers import *

# connection port
def build_model():
    model = LongFormer() # baseline
    return model

# Models Declaration
class LongFormer(nn.Module):
    def __init__(self, MODEL_NAME="allenai/longformer-base-4096", MAX_LEN=1024):
        self.MAX_LEN = MAX_LEN

        # pretrained model (Transformers)
        config = AutoConfig.from_pretrained(MODEL_NAME)
        backbone = AutoModel.from_pretrained(MODEL_NAME, config=config)
        
        # freeze or not the parameters for backbone
        for param in backbone.parameters():
            param.requires_grad = True # True for fine-tune, False for pre-train
        
        # output head
        self.out_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        )

        # loss function
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, input_pack):
        # unpack
        input_ids = input_pack['input_ids']
        mask = input_pack['attention_mask']

        # forward
        out = self.backbone(input_ids, attention_mask=mask)
        return self.out_fc(out)
    
    def loss(self, input_pack):
        # forwawrd
        out = self.forward(input_pack)

        # unpack and compute objective function
        labels = input_pack['labels']
        obj = self.loss_func(out, labels)
        return obj
    
    def predict(self, input_pack):
        if input_pack.get('labels') == None:
            y_true = list()
        else:
            y_true = input_pack['labels'].argmax(dim=2).squeeze()
        y_pred = self.forward(input_pack).argmax(dim=2).squeeze()
        return y_pred, y_true
