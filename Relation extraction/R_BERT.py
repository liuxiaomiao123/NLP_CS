# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 01:52:47 2024

@author: liangyingliu

note: I wrote this script by referring to the script on https://github.com/monologg/R-BERT/blob/master/trainer.py
"""

#%%

import config
import argparse
import os
import numpy as np
import model
from model import RobustBERT
from model import FullyConnectedLayer
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
import data_transformation
from config import parse_args
import math


#%%

args = config.parse_args()
relation_table, train_loader, dev_loader, test_loader = data_transformation.data_loader()

#%%

def evaluate(my_model, dataset, device, args):
    eval_dataloader = dataset

    labels_all = []
    predicted_labels_all = []

    my_model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            inputs = {
                "token_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "e1_attention_mask": batch[3],
                "e2_attention_mask": batch[4],
                "labels": batch[5]
            }
            outputs = my_model(**inputs)

            labels_all.extend(inputs["labels"].detach().cpu().numpy())
            predicted_labels_all.extend(outputs[2].detach().cpu().numpy())
   
    print('\n')
    print(f'predicted_labels_all: {predicted_labels_all} \n')
    print(f'labels_all: {labels_all} \n')
    
    
    return f1_score(np.array(labels_all), np.array(predicted_labels_all), average='weighted')


def validation(model, dev_loader, device, args):
    f1_validation = evaluate(model, dev_loader, device, args)
    print(f'Validation F1 Score: {f1_validation:.4f}')
    return f1_validation

def test(model, test_loader, device, args):
    f1_test = evaluate(model, test_loader, device, args)
    print(f'Test F1 Score: {f1_test:.4f}')
    return f1_test

#%%

num_relations = len(relation_table)
params = BertConfig.from_pretrained(args.model_name, num_labels = num_relations)
my_model = RobustBERT.from_pretrained(args.model_name, config = params, args = args)  # 不能与model.py重名

device = "cuda" if torch.cuda.is_available() else "cpu"
my_model.to(device)

#%%

if args.max_train_steps > 0:   # 我的错误在于python除了将 0，None, 空数据结构， False认为是False外，其余都是True. 所以-1的话这里是True.
    train_steps = args.max_train_steps
    args.num_epochs = (
        args.max_train_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
    )
else:
    train_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_epochs


#%%    

excluded_params = ["bias", "LayerNorm.weight"]   # no weight decay
optimizer_param_groups = [
    {
        "params": [param for name, param in my_model.named_parameters() 
                   if not any(exclude in name for exclude in excluded_params)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [param for name, param in my_model.named_parameters() 
                   if any(exclude in name for exclude in excluded_params)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(
    optimizer_param_groups,
    lr = args.lr,
    eps = args.adam_epsilon,
)

scheduler = get_linear_schedule_with_warmup(
           optimizer,
           num_warmup_steps = args.warmup_steps,
           num_training_steps = train_steps,
       )
#%%

train_loss = 0.0
f1_train = []
my_model.zero_grad()
    
for epoch in trange(int(args.num_epochs), desc="Epoch"):
    epoch_iterator = tqdm(train_loader, desc="Iteration")
    labels_ep = []
    predicted_labels_ep = []
    
    for batch_idx, batch in enumerate(epoch_iterator):
        my_model.train()
        batch = tuple(b.to(device) for b in batch)  

        inputs = {
          "token_ids": batch[0],
          "token_type_ids": batch[1],
          "attention_mask": batch[2],
          "e1_attention_mask": batch[3],
          "e2_attention_mask": batch[4],
          "labels": batch[5]
        }
        outputs = my_model(**inputs)  # 由于使用了**input，表示将字典中的键值对以关键字参数的形式传递给函数或方法，所以要保证这里的参数顺序与外面定义的一样
        loss = outputs[0]
        
        if math.isnan(loss):
            print('the bad data point index is : ')
            print(batch_idx)
            print('\n')
    
        loss.backward()
    
        train_loss += loss.item()
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), args.max_grad_norm)
    
            optimizer.step()
            scheduler.step()  
            my_model.zero_grad()
        
        labels_ep.extend(inputs["labels"].detach().cpu().numpy())
        predicted_labels_ep.extend(outputs[2].detach().cpu().numpy())
    
    print('\n')
    print(f'predicted_labels_ep: {predicted_labels_ep} \n')
    print(f'labels_ep: {labels_ep} \n')
    
    
    # 计算F1分数
    f1 = f1_score(np.array(labels_ep), np.array(predicted_labels_ep), average='weighted')
    print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {train_loss/len(train_loader):.4f}')
    print(f'Epoch {epoch}_Train F1 Score: {f1:.4f}')
    
    f1_train.append(f1)


#%%

f1_validation = validation(model = my_model, dev_loader = dev_loader, device = device, args = args)
f1_test = test(model = my_model, test_loader = test_loader, device = device, args = args)      

#%%

output_dir = r'/kaggle/working/'  
torch.save(my_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))  

