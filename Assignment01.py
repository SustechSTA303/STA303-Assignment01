#!/usr/bin/env python
# coding: utf-8

# # Assignment 01: Multi-class Classification 
# In this Assignment, you will train a deep model on the CIFAR10 from the scratch using PyTorch.

# ### Basic Imports

# In[33]:

import os
import time
import argparse
import os.path as osp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import models
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt
from PIL import Image


# ### Hyperparameters

# In[34]:
parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--lr', type=float, default=1e-1, help='lr')
parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss')
parser.add_argument('--model', type=str, default='vanilla', help='model')
parser.add_argument('--gamma', type=float, default=2, help='gamma')
args = parser.parse_args()


# random seed
SEED = 1 
NUM_CLASS = 10

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 60
EVAL_INTERVAL=1
SAVE_DIR = './log'

# Optimizer
LEARNING_RATE = args.lr
# LEARNING_RATE = 1e-2
# LEARNING_RATE = 1e-3
MOMENTUM = 0.9
STEP=5
GAMMA=0.5


# ### Device

# In[35]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 
# ### Dataset
# 

# In[36]:


# cifar10 transform
transform_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform_cifar10_train)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform_cifar10_test)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# ### Model

# In[37]:


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


# In[38]:

if(args.model=='vanilla'):
    model = ConvNet()
else:
    model=models.__dict__[args.model]()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
model.to(device)


# ### Optimizer

# In[39]:


optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)


# ### Task 1: per batch training/testing
# ---
# 
# Please denfine two function named ``train_batch`` and ``test_batch``. These functions are essential for training and evaluating machine learning models using batched data from dataloaders.
# 
# **To do**: 
# 1. Define the loss function i.e [nn.CrossEntropyLoss()](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
# 2. Take the image as the input and generate the output using the pre-defined SimpleNet.
# 3. Calculate the loss between the output and the corresponding label using the loss function.

# In[40]:


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda(0)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# In[41]:


##################### Write your answer here ##################
# Define the loss function
# criterion = nn.CrossEntropyLoss()
# criterion=nn.L1Loss()
# criterion=FocalLoss()
if(args.loss=='Focal'):
    criterion=FocalLoss(gamma=args.gamma,class_num=10)
elif(args.loss=='L1'):
    criterion=nn.L1Loss()
elif(args.loss=='CrossEntropy'):
    criterion=nn.CrossEntropyLoss()
###############################################################


# In[42]:


def train_batch(model, image, target):
    """
    Perform one training batch iteration.

    Args:
        model (torch.nn.Module): The machine learning model to train.
        image (torch.Tensor): Batch of input data (images).
        target (torch.Tensor): Batch of target labels.

    Returns:
        torch.Tensor: Model output (predictions) for the batch.
        torch.Tensor: Loss value calculated by the defined loss function loss_fn().
    """
    
    ##################### Write your answer here ##################
    output = model(image)
    # loss = criterion(output,target)
    if(args.loss=='L1'):
        loss = criterion(F.softmax(output, dim=1),F.one_hot(target,num_classes=10))
    elif(args.loss=='Focal' or args.loss=='CrossEntropy'):
        loss = criterion(output,target)
    ###############################################################

    return output, loss


# In[43]:


def test_batch(model, image, target):
    """
    Perform one testing batch iteration.

    Args:
        model (torch.nn.Module): The machine learning model to evaluate.
        image (torch.Tensor): Batch of input data (images).
        target (torch.Tensor): Batch of target labels.

    Returns:
        torch.Tensor: Model output (predictions) for the batch.
        torch.Tensor: Loss value calculated for the batch.
    """

    ##################### Write your answer here ##################
    output = model(image)
    # loss = criterion(output,target)
    # loss = criterion(F.softmax(output, dim=1),F.one_hot(target,num_classes=10))
    if(args.loss=='L1'):
        loss = criterion(F.softmax(output, dim=1),F.one_hot(target,num_classes=10))
    elif(args.loss=='Focal' or args.loss=='CrossEntropy'):
        loss = criterion(output,target)
    ###############################################################

    return output, loss


# ### Model Training

# In[44]:


training_loss = []
training_acc = []
testing_loss = []
testing_acc = []
lrs=[]

for epoch in range(NUM_EPOCHS):
    model.train()
    torch.cuda.empty_cache()

    ##########################
    ### Training
    ##########################

    running_cls_loss = 0.0
    running_cls_corrects = 0

    for batch_idx, (image, target) in enumerate(train_dataloader):

        image = image.to(device)
        target = target.to(device)

        # train model
        outputs, loss = train_batch(model, image, target)
        _, preds = torch.max(outputs, 1)

        
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')
        running_cls_loss += loss.item()
        running_cls_corrects += torch.sum(preds == target.data)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # epoch_loss = running_cls_loss / len(train_set)
    epoch_loss = running_cls_loss / len(train_dataloader)
    epoch_acc = running_cls_corrects.double() / len(train_set)

    print(f'Epoch: {epoch+1}/{NUM_EPOCHS} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    training_loss.append(epoch_loss)
    training_acc.append(epoch_acc.cpu().detach().numpy())

    # change learning rate
    scheduler.step()


    ##########################
    ### Testing
    ##########################
    # # eval model during training or in the last epoch
    if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch +1) == NUM_EPOCHS:
        print('Begin test......')
        model.eval()
    
        val_loss = 0.0
        val_corrects = 0

        for batch_idx, (image, target) in enumerate(test_dataloader):

            image = image.to(device)
            target = target.to(device)

            # test model
            outputs, loss = test_batch(model, image, target)
            _, preds = torch.max(outputs, 1)
            
            val_loss += loss.item()
            val_corrects += torch.sum(preds == target.data)

        val_loss = val_loss / len(test_dataloader)
        val_acc = val_corrects.double() / len(test_set)
        print(f'Test Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        testing_loss.append(val_loss)
        testing_acc.append(val_acc.cpu().detach().numpy())

        # save the model in last epoch
        if (epoch +1) == NUM_EPOCHS:
            
            state = {
            'state_dict': model.state_dict(),
            'acc': epoch_acc,
            'epoch': (epoch+1),
            }

            # check the dir
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)

            # save the state
            torch.save(state, osp.join(SAVE_DIR, 'checkpoint_%s.pth' % (str(epoch+1))))

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    lrs.append(lr)


# In[ ]:

target_path=f"save/{args.model}"
if not os.path.exists(target_path):
    os.makedirs(target_path)
np.save(os.path.join(target_path, f"{args.loss}_{args.lr}_{args.gamma}_trainloss.npy"),training_loss)
np.save(os.path.join(target_path, f"{args.loss}_{args.lr}_{args.gamma}_testloss.npy"),testing_loss)
np.save(os.path.join(target_path, f"{args.loss}_{args.lr}_{args.gamma}_trainacc.npy"),training_acc)
np.save(os.path.join(target_path, f"{args.loss}_{args.lr}_{args.gamma}_testacc.npy"),testing_acc)
np.save(os.path.join(target_path, f"{args.loss}_{args.lr}_{args.gamma}_lr.npy"),lrs)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(range(NUM_EPOCHS), training_loss, label='Training Loss', color='blue')
plt.plot(range(NUM_EPOCHS), testing_loss, label='Testing Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')

# 绘制训练与测试的准确度随epoch的变化
plt.subplot(132)
plt.plot(range(NUM_EPOCHS), training_acc, label='Training Accuracy', color='blue')
plt.plot(range(NUM_EPOCHS), testing_acc, label='Testing Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Testing Accuracy')

# 绘制学习率随epoch的变化
plt.subplot(133)
plt.plot(range(NUM_EPOCHS), lrs, label='Learning Rate', color='green')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate')

plt.tight_layout()
plt.savefig(os.path.join(target_path, f"{args.loss}_{args.lr}_{args.gamma}.png"))


# ### Task 2: Instance inference
# ---
# The task is to visualizes an image along with model prediction and class probabilities.
# 
# **To do**: 
# 1. Calculate the prediction and the probabilities for each class.
#          

# In[ ]:


# inputs, classes = next(iter(test_dataloader))
# input = inputs[0]


# In[ ]:


##################### Write your answer here ##################
# input: image, model
# outputs: predict_label, probabilities
# predict_label is the index (or label) of the class with the highest probability from the probabilities.
###############################################################
# input=input.to(device)
# m=nn.Softmax(dim=1)
# probabilities = torch.squeeze(m(model(input.to(device))))
# predict_label = torch.argmax(probabilities)


# In[ ]:


# predicted_class = class_names[predict_label.item()]
# predicted_probability = probabilities[predict_label].item()
# image = input.cpu().numpy().transpose((1, 2, 0))
# plt.imshow(image)
# plt.text(17, 30, f'Predicted Class: {predicted_class}\nProbability: {predicted_probability:.2f}', 
#             color='white', backgroundcolor='black', fontsize=8)
# plt.show()

# # Print probabilities for each class
# print('Print probabilities for each class:')
# for i in range(len(class_names)):
#     print(f'{class_names[i]}: {probabilities[i].item():.4f}')

