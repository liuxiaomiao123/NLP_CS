# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:54:00 2024

@author: liangyingliu

note: I wrote this model by referring to the script on https://github.com/monologg/R-BERT/blob/master/model.py
"""

#%%

import config
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob, activate = True):
        super(FullyConnectedLayer, self).__init__()
        self.activate = activate
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc_layer = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        x = self.dropout_layer(x)
        if self.activate:
            x = nn.Tanh()
        return self.fc_layer(x)

class RobustBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(RobustBERT, self).__init__(config)
        self.bert_model = BertModel(config = config)
        self.CLS_embed = FullyConnectedLayer(config.hidden_size, config.hidden_size, args.dropout_prob)
        self.entity_embed = FullyConnectedLayer(config.hidden_size, config.hidden_size, args.dropout_prob)
        self.label_count = config.num_labels
        self.ouput = FullyConnectedLayer(config.hidden_size*3, config.num_labels, args.dropout_prob, activate = False)
     
     
    @staticmethod
    def avg_entity(hidden_states, entity_mask):
        vec = torch.bmm(entity_mask.unsqueeze(1).float(), hidden_states).squeeze(1)
        vec_avg = vec.float() / (entity_mask != 0).sum(dim=1).unsqueeze(1).float() 
        return vec_avg
    
    
    # 由于使用了**input，表示将字典中的键值对以关键字参数的形式传递给函数或方法，所以要保证这里的参数顺序与外面定义的一样
    def forward(self, token_ids, token_type_ids, attention_mask, e1_attention_mask, e2_attention_mask, labels):
        bert_out = self.bert_model(
            token_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ) 
        seq = bert_out[0]
        cls_out = bert_out[1]  
        entity1_hidden = self.avg_entity(seq, e1_attention_mask)
        entity2_hidden = self.avg_entity(seq, e2_attention_mask)
        entity1_hidden = self.entity_embed(entity1_hidden)
        entity2_hidden = self.entity_embed(entity2_hidden)
        cls_out = self.CLS_embed(cls_out)
        concatenated_hidden = torch.cat([cls_out, entity1_hidden, entity2_hidden], dim=-1)
        logits = self.output(concatenated_hidden)
        
        loss_function = nn.CrossEntropyLoss()
        loss_value = loss_function(logits.view(-1, self.label_count), labels.view(-1))
        _, predicted_labels = torch.max(logits, dim=1)
        out = (loss_value,) + (logits,) + (predicted_labels,) + bert_out[2:]       
            
        return out  
        
        