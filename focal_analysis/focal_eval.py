# Basic imports
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import argparse

from torchvision import transforms
import torchvision
from torchvision.models import resnet18

import random

from baseline_model import ConvNet
from focal import FocalLoss


## Hyperparameters
# random seed
SEED = 1
NUM_CLASS = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(SEED)

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 50
EVAL_INTERVAL = 1
SAVE_DIR = './log'

# Optimizer
LEARNING_RATE = 1e-1
MOMENTUM = 0.9
STEP = 5
GAMMA = 0.5

## Parse arguments
parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
parser.add_argument("--gamma", type=float, default="2", help="Gamma") # 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2
parser.add_argument("--model", type=str, default="baseline", help="Model")  # baseline resnet18
super_args = parser.parse_args()
LEARNING_RATE = super_args.lr

## Device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

## Dataset
match super_args.model:
    case "basline":
        size = 32
    case "resnet18":
        size = 224
    case _:
        size = 32

transform_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_cifar10_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                        download=True, transform=transform_cifar10_train)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                       download=True, transform=transform_cifar10_test)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


## Init model
match super_args.model:
    case "baseline":
        model = ConvNet()
        model.to(device)
    case "resnet18":
        model = resnet18(num_classes=10)
        model.to(device)
    case _:
        raise Exception("Wrong model argument")

## Init optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)

## Init Loss
criterion = FocalLoss(gamma=super_args.gamma, reduction="mean")

def train_batch(model, image, target):
    output = model(image)
    loss = criterion(output, target)  
    return output, loss  

def test_batch(model, image, target):
    output = model(image)
    loss = criterion(output, target)
    return output, loss

## Train model
test_loss_l = []
test_acc_l = []
test_precision_l = []
test_recall_l = []
test_f1score_micro_l = []
test_f1score_macro_l = []

test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)
test_precision = torchmetrics.classification.MulticlassPrecision(num_classes=10, average='macro').to(device)
test_recall = torchmetrics.classification.MulticlassRecall(num_classes=10, average='macro').to(device)
test_f1score_micro = torchmetrics.classification.MulticlassF1Score(num_classes=10, average='micro').to(device)
test_f1score_macro = torchmetrics.classification.MulticlassF1Score(num_classes=10, average='macro').to(device)

for epoch in range(NUM_EPOCHS):
    model.train()
    torch.cuda.empty_cache()

    for batch_idx, (image, target) in enumerate(train_dataloader):
        image = image.to(device)
        target = target.to(device)

        outputs, loss = train_batch(model, image, target)
        _, preds = torch.max(outputs, 1)

        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")

    scheduler.step()

    if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
        print("Begin test......")
        model.eval()

        test_loss = .0
        test_acc.reset()
        test_precision.reset()
        test_recall.reset()
        test_f1score_micro.reset()
        test_f1score_macro.reset()

        for batch_idx, (image, target) in enumerate(test_dataloader):
            
            image = image.to(device)
            target = target.to(device)

            # Test model
            outputs, loss = test_batch(model, image, target)
            _, preds = torch.max(outputs, 1)
            
            test_loss += loss.item()
            test_acc.update(preds, target)
            test_precision.update(preds, target)
            test_recall.update(preds, target)
            test_f1score_micro.update(preds, target)
            test_f1score_macro.update(preds, target)

        val_loss = test_loss / len(test_set)
        val_acc = test_acc.compute()
        val_precision = test_precision.compute()
        val_recall = test_recall.compute()
        val_f1score_micro = test_f1score_micro.compute()
        val_f1score_macro = test_f1score_macro.compute()
        print(f'Test Loss: {val_loss:.4f} Acc: {val_acc:.4f} Precision: {val_precision:.4f} Recall: {val_recall:.4f} f1score_micro: {val_f1score_micro:.4f} f1score_macro: {val_f1score_macro:.4f}')

        test_loss_l.append(test_loss)
        test_acc_l.append(val_acc.cpu().detach().numpy())
        test_precision_l.append(val_precision.cpu().detach().numpy())
        test_recall_l.append(val_recall.cpu().detach().numpy())
        test_f1score_micro_l.append(val_f1score_micro.cpu().numpy())
        test_f1score_macro_l.append(val_f1score_macro.cpu().numpy())

df = pd.DataFrame({'acc': test_acc_l, 'precision': test_precision_l, 'recall': test_recall_l, 'f1score_micro': test_f1score_micro_l, 'f1score_macro': test_f1score_macro_l})
df.to_csv(f'./log/{super_args.model}_focal{super_args.gamma}.csv', index=False)
print('log generated')