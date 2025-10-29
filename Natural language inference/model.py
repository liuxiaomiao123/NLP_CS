# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:21:42 2024

@author: liangyingliu
"""


import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BertNLI(nn.Module):
    def __init__(self, num_labels = 3):
        super(BertNLI, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)  # 768
        # 也可以写成 self.config = BertConfig.from_pretrained("bert-base-cased"),  self.config.hidden_size
        
        
    def forward(self, x):
        bert_outputs = self.bert(input_ids = x[0], token_type_ids = x[1], attention_mask = x[2])
        cls_out = bert_outputs[1]
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out)
        
        loss_function = nn.CrossEntropyLoss()
        labels_true = x[3]
        loss_value = loss_function(logits.view(-1, self.num_labels), labels_true.view(-1))
        _, labels_predicted = torch.max(logits, dim=1)
        
        out = (loss_value,) + (logits,) + (labels_predicted,) + bert_outputs[2:]
        
        return out
