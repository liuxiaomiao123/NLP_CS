# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:39:44 2024

@author: liangyingliu
"""

import argparse


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'bert-base-uncased')
    parser.add_argument('--max_seq_len', type = int, default = 300)
    parser.add_argument('--train_batch_size', type = int, default = 4)
    parser.add_argument('--dev_batch_size', type = int, default = 16)
    parser.add_argument('--test_batch_size', type = int, default = 16)
    parser.add_argument('--eval_batch_size', type = int, default = 4)
    parser.add_argument('--num_epochs', type = int, default = 2)
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1)
    parser.add_argument('--max_train_steps', type = int, default = 1)
    parser.add_argument('--weight_decay', type = float, default = 0.0)
    parser.add_argument('--lr', type = float, default = 2e-5)
    parser.add_argument('--adam_epsilon', type = float, default = 1e-8)
    parser.add_argument('--warmup_steps', type = int, default = 0)
    parser.add_argument("--max_grad_norm", type = float, default = 1.0)
    parser.add_argument("--dropout_prob", type = float, default = 0.1)

    args = parser.parse_args()
    return args

