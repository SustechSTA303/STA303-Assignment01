import os
import time
import os.path as osp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Subset,ConcatDataset

from torchvision import datasets
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt
from PIL import Image
import time
from models import *
from utility import *
from train import train
from test import test
import shutil
import pickle
import json
# from models.resnet import ResNet18
#更改
from plot import Plot
import torchvision.models as models
"""### Hyperparameters"""

# random seed
SEED = 1
NUM_CLASS = 10

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 100
EVAL_INTERVAL=1
SAVE_DIR = './log'

# Optimizer
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
STEP=5
GAMMA=0.5

"""### Device"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

"""
### Dataset
"""


# cifar10 transform


# 定义 CIFAR-10 数据的数据变换
transform_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载原始 CIFAR-10 数据集
cifar10_train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_cifar10_train)

# 定义每个类别的目标样本数量
target_class_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

imbalance_datasets = []

for class_label in range(10):
    num_samples = target_class_samples[class_label]

    # 选择每个类别的前 num_samples 个样本
    class_indices = [i for i, (data, label) in enumerate(cifar10_train) if label == class_label]
    selected_indices = class_indices[:num_samples]

    # 创建子集
    subset = Subset(cifar10_train, selected_indices)
    imbalance_datasets.append(subset)

# 创建一个不平衡的训练数据集
train_set = ConcatDataset(imbalance_datasets)

# 设置批量大小和数据加载器
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)




transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# train_set = torchvision.datasets.CIFAR10(root='../data', train=True,
#                                          download=True, transform=transform_cifar10_train)
# train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
#                                                shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform_cifar10_test)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.fc1 = nn.Linear(8 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# LossFun
loss_functions = [
    (nn.CrossEntropyLoss(), 'CrossEntropyLoss'),
    (nn.L1Loss(), "L1Loss"),
    (FocalLoss(gamma=0.5), "FocalLoss(Gamma=0.5)"),
    (FocalLoss(gamma=2), "FocalLoss(Gamma=2)"),

]
results = {}


# 训练
for criterion, criterion_name in loss_functions:
    
    print(criterion_name)

#     model = ConvNet()
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASS)
    
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    result_data = {
        'training_loss': [],
        'training_acc': [0,],
        'training_times': [0,],
        'training_weighted_avg_precision': [0,],
        'training_weighted_avg_recall': [0,],
        'training_weighted_avg_f1_score': [0,],
        #更改
        'training_confusion':[],
        
        'testing_loss': [],
        'testing_acc': [0,],
        'testing_times': [0,],
        'testing_weighted_avg_precision': [0,],
        'testing_weighted_avg_recall': [0,],
        'testing_weighted_avg_f1_score': [0,],
        #更改
        'testing_confusion':[],
        
        
        'total_training_time': 0.0,
        'total_testing_time': 0.0
        
    }

    total_training_time = 0.0
    total_testing_time = 0.0
    training_times = []
    testing_times = []
    
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc, epoch_training_time, report, confusion = train(model, train_dataloader, criterion, optimizer, scheduler, device, class_names, train_set)
        total_training_time += epoch_training_time
        print(f'Epoch: {epoch + 1}/{NUM_EPOCHS} Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        scheduler.step()
        
        training_weighted_avg_precision = report['weighted avg']['precision']
        training_weighted_avg_recall = report['weighted avg']['recall']
        training_weighted_avg_f1_score = report['weighted avg']['f1-score']
       
        
        result_data['training_loss'].append(train_loss)
        result_data['training_acc'].append(train_acc)
        result_data["training_times"].append(total_training_time)
        result_data['total_training_time']=total_training_time
        
        result_data['training_weighted_avg_precision'].append(training_weighted_avg_precision)
        result_data['training_weighted_avg_recall'].append(training_weighted_avg_recall)
        result_data['training_weighted_avg_f1_score'].append(training_weighted_avg_f1_score)
        #更改
        result_data['training_confusion'].append(confusion.tolist())

        # Eval
        if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            test_loss, test_acc, epoch_testing_time, report,confusion = test(model, test_dataloader, criterion, device, class_names, test_set)
            total_testing_time+=epoch_testing_time
            
            #更改
            print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
            
            testing_weighted_avg_precision = report['weighted avg']['precision']
            testing_weighted_avg_recall = report['weighted avg']['recall']
            testing_weighted_avg_f1_score = report['weighted avg']['f1-score']

            result_data['testing_loss'].append(test_loss)
            result_data['testing_acc'].append(test_acc)
            result_data['testing_times'].append(total_testing_time)
            result_data['total_testing_time']=total_testing_time
            result_data['testing_weighted_avg_precision'].append(testing_weighted_avg_precision)
            result_data['testing_weighted_avg_recall'].append(testing_weighted_avg_recall)
            result_data['testing_weighted_avg_f1_score'].append(testing_weighted_avg_f1_score)
            #更改
            result_data['testing_confusion'].append(confusion.tolist())
            # save the model in the last epoch
            if (epoch + 1) == NUM_EPOCHS:
                state = {
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'epoch': (epoch + 1),
                }

                # check the dir
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)

                # save the state
                torch.save(state, osp.join(SAVE_DIR, f'checkpoint_{criterion_name}_{epoch + 1}.pth'))

    # 将每种损失函数的结果存储在字典中，以损失函数的名称作为键
    results[criterion_name] = result_data

# print(results)


# 保存 results 到文件
if not os.path.exists("record"):
    os.makedirs("record")
with open('record/results19.json', 'w') as file:
    json.dump(results, file)

    

# 画图
# 更改
# Plot(NUM_EPOCHS, 'record/results15.json', 'images/images15')


# # size
# sample_sizes = [500, 2500, 5000, 10000, 50000]
# recur = 3
# results_size_train = {}
# results_size_test = {}


# for criterion, criterion_name in loss_functions:
#     training_accs = [0,]
#     testing_accs = [0,]
#     print(criterion_name)
#     for sample_size in sample_sizes:
#         print("sample_size:", sample_size)
#         results_size_test[criterion_name] = {}
#         subset_indices = torch.randperm(len(train_set))[:sample_size]
#         train_subset = torch.utils.data.Subset(train_set, subset_indices)
#         train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#         subset_indices = torch.randperm(len(test_set))[:sample_size]
#         test_subset = torch.utils.data.Subset(test_set, subset_indices)
#         test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#         aux_training_acc =[]
#         aux_testing_acc = []
#         for i in range(recur):
#             print(i+1)
#             model = ConvNet()
#             model.to(device)
#             optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
#             for epoch in range(NUM_EPOCHS):
#                 train_loss, train_acc, epoch_training_time = train(model, train_dataloader, criterion, optimizer, scheduler, device, class_names, train_set)
#                 print(f'Epoch: {epoch + 1}/{NUM_EPOCHS} Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
#                 scheduler.step()
#                 if epoch+1 == NUM_EPOCHS:
#                     aux_training_acc.append(train_acc)
#                     print("aux_training_acc:",aux_training_acc)
#             ##test
#             if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
#                 sample_size/=5
#                 test_loss, test_acc, epoch_testing_time = test(model, test_dataloader, criterion, device, class_names, test_set)
#                 print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
#                 aux_testing_acc.append(test_acc)
#                 print("aux_testing_acc", aux_testing_acc)
#         training_accs.append(sum(aux_training_acc) / recur)
#         testing_accs.append(sum(aux_testing_acc) / recur)
#         print("training_accs_avg:",training_accs)
#         print("testing_accs_avg:",testing_accs)
#     results_size_train[criterion_name] = training_accs
#     print("results_size_train",results_size_train)
#     results_size_test[criterion_name] = testing_accs
#     print("results_size_test",results_size_test)



#
# """### Task 2: Instance inference
# ---
# The task is to visualizes an image along with model prediction and class probabilities.
#
# **To do**:
# 1. Calculate the prediction and the probabilities for each class.
#
# """
#
# inputs, classes = next(iter(test_dataloader))
# input = inputs[0]
# input = input.to(device)
#
# ##################### Write your answer here ##################
# # input: image, model
# # outputs: predict_label, probabilities
# # predict_label is the index (or label) of the class with the highest probability from the probabilities.
# ###############################################################
# print(model(input))
# print(model(input).softmax(dim=1))
# print(model(input).softmax(dim=1).squeeze())
# probabilities = model(input).softmax(dim=1).squeeze()
# predict_label = probabilities.argmax()
#
# predicted_class = class_names[predict_label.item()]
# predicted_probability = probabilities[predict_label].item()
# image = input.cpu().numpy().transpose((1, 2, 0))
# plt.imshow(image)
# plt.text(17, 30, f'Predicted Class: {predicted_class}\nProbability: {predicted_probability:.2f}',
#          color='white', backgroundcolor='black', fontsize=8)
# plt.show()
#
# # Print probabilities for each class
# print('Print probabilities for each class:')
# for i in range(len(class_names)):
#     print(f'{class_names[i]}: {probabilities[i].item():.4f}')

