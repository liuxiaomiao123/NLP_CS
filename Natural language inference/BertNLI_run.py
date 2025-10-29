# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:46:16 2024

@author: liangyingliu
"""

import Data_transformation
from model import BertNLI
import os
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm, trange


train_loader, dev_loader, test_loader = Data_transformation.transform()

def evaluate(my_model, dataset, device):
    eval_dataloader = dataset
    
    labels = []
    labels_predicted = []
    
    my_model.eval()
    
    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        batch = tuple(b.to(device) for b in batch)  # transfer data from cpu to gpu
        
        with torch.no_grad():
            outputs = my_model(batch)
            
            labels.extend(batch[3].detach().cpu().numpy())
            labels_predicted.extend(outputs[2].detach().cpu().numpy())

    correct_predictions = np.sum(np.array(labels) == np.array(labels_predicted))
    accuracy = correct_predictions / len(labels)
    
    return accuracy


def validation(model, dev_loader, device):
    acc_val = evaluate(model, dev_loader, device)
    print(f'Validation ACC Score: {acc_val:.4f}')
    return acc_val


def test(model, test_loader, device):
    acc_test = evaluate(model, test_loader, device)
    print(f'Test ACC Score: {acc_test:.4f}')
    return acc_test


my_model = BertNLI()
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device('cpu')
my_model.to(device)

optimizer = Adam(my_model.parameters(), lr = 2e-5, eps=1e-6)

num_epochs = 3

acc_train = []

for epoch in trange(num_epochs, desc = "Epoch"):
    labels_ep = []
    labels_ep_predicted = []
    
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc = "Batch")):
        my_model.train()
        batch = tuple(b.to(device) for b in batch)  # transfer data from cpu to gpu
        
        outputs = my_model(batch)
        loss = outputs[0]
        
        loss.backward()  # 在调用 loss.backward() 后，PyTorch 会将所有的梯度存储在模型参数的 .grad 属性中。你可以通过 model.parameters() 访问这些梯度
        optimizer.step()  # 反向传播会累积梯度，如果不清除梯度（optimizer.zero_grad() 或 model.zero_grad()），它会在每次 backward() 调用时累加
        # 更新模型的参数（即梯度下降）
        # 在调用 loss.backward() 后，你已经计算出了梯度，但这时模型的参数并没有更新。optimizer.step() 通过使用这些梯度来调整模型的参数，使损失函数的值逐步减小。
        my_model.zero_grad()
        
        labels_ep.extend(batch[3].detach().cpu().numpy())
        labels_ep_predicted.extend(outputs[2].detach().cpu().numpy())
        
    correct_predictions = np.sum(np.array(labels_ep) == np.array(labels_ep_predicted))
    accuracy = correct_predictions / len(labels_ep)
    print(f'Epoch {epoch}_Train ACC Score: {accuracy:.4f}')
    
    acc_train.append(accuracy)

print("\nAll Epochs Accuracy:")
for epoch, acc in enumerate(acc_train, start=1):
    print(f"Epoch {epoch}: {acc:.4f}")


acc_validation = validation(model = my_model, dev_loader = dev_loader, device = device)
acc_test = test(model = my_model, test_loader = test_loader, device = device)  
    

output_dir = '/home/liangyingliu/HW3/AllNLI'
torch.save(my_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))



#with open("acc_train.txt", "w") as file:
#    file.write("\n".join(map(str, acc_train)))

#with open("acc_validation.txt", "w") as file:
#    file.write("\n".join(map(str, acc_validation)))

#with open("acc_test.txt", "w") as file:
#    file.write("\n".join(map(str, acc_test)))



