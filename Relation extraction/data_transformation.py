# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:33:00 2024

@author: liangyingliu
"""

#%%

import os
import numpy as np
import pandas as pd
import re

import argparse
from transformers import BertTokenizer, BertConfig

import torch
from torch.utils.data import TensorDataset   
from config import parse_args
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

args = parse_args()

#%%

def data_comb(sentence_path, label_path):
    
    sentences = []
    
    with open(sentence_path, 'r', encoding = 'utf-8') as file:
        for sid, line in enumerate(file):
            sent = line.strip()
            sentences.append([sid, sent])
    df_sentences = pd.DataFrame(sentences, columns = ['sid', 'sentence'])
    
    
    tuples = []
    with open(label_path, 'r', encoding = 'utf-8') as file:
        for sid, line in enumerate(file):
            tuple_l = line.strip().split('|')
            for tuple_i in tuple_l:
                parts = tuple_i.strip().split(';')
                e1 = parts[0].strip()
                e2 = parts[1].strip()
                relation = parts[2].strip()
                
                tuples.append([sid, e1, e2, relation])
    df_tuples = pd.DataFrame(tuples, columns = ['sid', 'e1' , 'e2', 'relation'])
        
    df_cb = pd.merge(df_sentences, df_tuples, on = 'sid', how = 'inner')
    
    return df_cb



def  data_input(df, relation_table, tokenizer, args):
    
    input_list = []
    
    for idx, row in df.iterrows():
        sent = row['sentence']
        sent = sent.translate(str.maketrans('', '', '#$'))   # 需要去除标记e1和e2的特殊字符吗？
        sent_tokens = tokenizer.tokenize(sent)   # 由于BERT这种大模型是在海量数据上训练的，所以保留标点符号是有用的
        
        sent_tokens = ['CLS'] + sent_tokens  #加上CLS
        
        e1 = tokenizer.tokenize(row['e1'])
        e2 = tokenizer.tokenize(row['e2'])
                
        
        e1_start_p = sent_tokens.index(e1[0])
        sent_tokens.insert(e1_start_p, '$')  
        e1_end_p = sent_tokens.index(e1[-1]) + 1
        sent_tokens.insert(e1_end_p, '$')    # 注意，这里有个很巧妙的陷阱，如果e1在e2的后面，那么这里的e1的index会随着e2 token的增加而改变
        
        e2_start_p = sent_tokens.index(e2[0])
        sent_tokens.insert(e2_start_p, '#')
        e2_end_p = sent_tokens.index(e2[-1]) + 1
        sent_tokens.insert(e2_end_p, '#')
        
        # 等所有special tokens都添加到列表后，再重新寻找index
        e1_start_p = sent_tokens.index(e1[0]) - 1
        e1_end_p = sent_tokens.index(e1[-1]) + 1
        e2_start_p = sent_tokens.index(e2[0]) - 1
        e2_end_p = sent_tokens.index(e2[-1]) + 1
        
        sent_tokens = sent_tokens + ['SEP']
        
        # truncuate
        if len(sent_tokens) > args.max_seq_len:
            sent_tokens = sent_tokens[: args.max_seq_len]
            
            
        attention_mask = [1] * len(sent_tokens)
        
        token_ids = tokenizer.convert_tokens_to_ids(sent_tokens)  # {#:1001, $:1002, CLS:100, SEP:100} 这可能也是为什么原文用#和$的原因
        
        padding_len = args.max_seq_len - len(sent_tokens)
        token_ids = token_ids + [0] * padding_len   # zero padding
        
        token_type_ids = [0] * len(token_ids)
        attention_mask = attention_mask + [0] * padding_len
        
        e1_attention_mask = [0] * len(token_ids)
        e2_attention_mask = [0] * len(token_ids)
        
        e1_attention_mask[e1_start_p:e1_end_p + 1] = [1] * (e1_end_p - e1_start_p + 1)
        e2_attention_mask[e2_start_p:e2_end_p + 1] = [1] * (e2_end_p - e2_start_p + 1)
        
        token_relation = relation_table.index(row['relation'])
        
        input_list.append([token_ids, token_type_ids, attention_mask, e1_attention_mask, e2_attention_mask, token_relation])
        
    df_input = pd.DataFrame(input_list, columns = ['token_ids', 'token_type_ids', 'attention_mask', 'e1_attention_mask', 'e2_attention_mask', 'token_relation'])
    return df_input


def df_to_tensor(df):
    token_ids = torch.tensor(df['token_ids'].tolist(), dtype = torch.long)
    token_type_ids = torch.tensor(df['token_type_ids'].tolist(), dtype = torch.long)
    attention_mask = torch.tensor(df['attention_mask'].tolist(), dtype = torch.long)
    e1_attention_mask = torch.tensor(df['e1_attention_mask'].tolist(), dtype = torch.long)
    e2_attention_mask = torch.tensor(df['e2_attention_mask'].tolist(), dtype = torch.long)
    token_relation = torch.tensor(df['token_relation'].tolist(), dtype = torch.long)
    
    tensor_input = TensorDataset(token_ids, token_type_ids, attention_mask, e1_attention_mask, e2_attention_mask, token_relation)
    
    return tensor_input

#%%

#max(df_t['sentence'].str.split().str.len())


#%%

tokenizer = BertTokenizer.from_pretrained(args.model_name)   

#%%

root_path = r'C:\Users\90542\OneDrive - Virginia Tech\Class_Slides\NLP\HW\NYT29-20231009T191130Z-001\NYT29'
train_sent_path = os.path.join(root_path, 'train.sent')
train_tuple_path = os.path.join(root_path, 'train.tup')

dev_sent_path = os.path.join(root_path, 'dev.sent')
dev_tuple_path = os.path.join(root_path, 'dev.tup')

test_sent_path = os.path.join(root_path, 'test.sent')
test_tuple_path = os.path.join(root_path, 'test.tup')

df_train = data_comb(train_sent_path, train_tuple_path)
df_dev = data_comb(dev_sent_path, dev_tuple_path)
df_test = data_comb(test_sent_path, test_tuple_path)

#%%

df_train_all = df_train
df_dev_all = df_dev
df_test = df_test

#%% demo testing

df_train = df_train_all.head(10)
df_dev = df_dev_all.head(10)
df_test = df_test.head(10)

#%%

relation_table_path = os.path.join(root_path, 'relations.txt')

with open(relation_table_path, 'r', encoding = 'utf-8') as file:
     relation_table = [line.strip() for line in file]
    
#%%

df_train_input = data_input(df_train, relation_table, tokenizer, args) 
df_dev_input = data_input(df_dev, relation_table, tokenizer, args) 
df_test_input = data_input(df_test, relation_table, tokenizer, args) 
    
#%%
    
train_dataset_input = df_to_tensor(df_train_input)
dev_dataset_input = df_to_tensor(df_dev_input)
test_dataset_input = df_to_tensor(df_test_input)

#%%

train_sampler = RandomSampler(train_dataset_input)
train_loader = DataLoader(train_dataset_input, sampler = train_sampler, batch_size = args.train_batch_size)

dev_sampler = RandomSampler(dev_dataset_input)
dev_loader = DataLoader(dev_dataset_input, sampler = dev_sampler, batch_size = args.eval_batch_size)

test_sampler = RandomSampler(test_dataset_input)
test_loader = DataLoader(test_dataset_input, sampler = test_sampler, batch_size = args.eval_batch_size)

