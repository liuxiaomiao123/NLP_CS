# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:50:56 2024

@author: liangyingliu
"""


import pandas as pd
from transformers import AutoTokenizer
import sys
import pickle
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import os

# s1_data_file = sys.argv[1]  # sys.argv[0]是脚本名，从sys.argv[1]开始是传入的参数
# s2_data_file = sys.argv[2]


def sent_comb(s1_data_file, s2_data_file):
    
    with open(s1_data_file, "r",  encoding = "utf-8") as f:
        s1_data = f.readlines()
        
    with open(s2_data_file, "r",  encoding = "utf-8") as f:
        s2_data = f.readlines()
        
    s1 = [sent.strip() for sent in s1_data]
    s2 = [sent.strip() for sent in s2_data]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    s = tokenizer(
        s1,
        s2,
        padding="max_length",          
        truncation=True,       
        max_length=300,        
        return_tensors="pt",   
    )
    
    return s

# s.input_ids  |   s.token_type_ids   |   s.attention_mask


def read_data(file):
    
    with open(file, "r", encoding = "utf-8") as f:
        data = f.readlines()
        
    data = [sent.strip() for sent in data]
    
    return data


def labels_to_ids(labels, dict):
    labels_ids = [dict[label] for label in labels]
    
    return labels_ids


def data_to_TensorDataset(data, labels, split = False):
    token_ids = data.input_ids
    token_type_ids = data.token_type_ids
    attention_mask = data.attention_mask
                                  
    labels_ids = torch.tensor(labels, dtype = torch.long)
    
    data_TensorDataset = TensorDataset(token_ids, token_type_ids, attention_mask, labels_ids)
    
    # demo testing
#    data_TensorDataset = TensorDataset(token_ids[:5000], token_type_ids[:5000], attention_mask[:5000], labels_ids[:5000])
    
    if split == True:
        train_TensorDataset, dev_TensorDataset = random_split(data_TensorDataset, lengths = [0.9, 0.1])
        return train_TensorDataset, dev_TensorDataset
    else:
        return data_TensorDataset
    


def data_loader(train_TensorDataset, dev_TensorDataset, test_TensorDataset):
    train_loader = DataLoader(train_TensorDataset, batch_size = 16, shuffle = True)
    dev_loader = DataLoader(dev_TensorDataset, batch_size = 16, shuffle = False)
    test_loader = DataLoader(test_TensorDataset, batch_size = 8, shuffle = False)
    
    return train_loader, dev_loader, test_loader



def transform():
    path_root = r"/home/liangyingliu/HW3/AllNLI"
    #path_root = r"C:\Users\90542\OneDrive - Virginia Tech\Class_Slides\NLP\HW\HW3\AllNLI\AllNLI"
    data = sent_comb(os.path.join(path_root, "s1.train"), os.path.join(path_root, "s2.train"))
    test_data = sent_comb(os.path.join(path_root, "s1.dev"), os.path.join(path_root, "s2.dev"))

    data_labels = read_data(os.path.join(path_root, "labels.train"))
    test_labels = read_data(os.path.join(path_root, "labels.dev"))

    labels_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}

    data_labels_id = labels_to_ids(data_labels, labels_dict)
    test_labels_id = labels_to_ids(test_labels, labels_dict)

    train_TensorDataset, dev_TensorDataset = data_to_TensorDataset(data, data_labels_id, split = True)
    test_TensorDataset = data_to_TensorDataset(test_data, test_labels_id, split = False)

    train_loader, dev_loader, test_loader = data_loader(train_TensorDataset, dev_TensorDataset, test_TensorDataset)
    
    return train_loader, dev_loader, test_loader 



# #%%
# # 获取 Input IDs
# input_ids = encoded.input_ids

# # 转换回 Tokens
# tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

# # 输出 Tokens
# for i, token_list in enumerate(tokens):
#     print(f"Sentence Pair {i + 1}:")
#     print(token_list)
