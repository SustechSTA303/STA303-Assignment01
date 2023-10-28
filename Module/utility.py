import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets
from torchvision import transforms
import torchvision

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.gather(1, target.view(-1, 1))
        alpha = self.alpha
        loss = -alpha * (1 - pt) ** self.gamma * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_batch(model, image, target, criterion):
    output = model(image)
    if isinstance(criterion, nn.L1Loss):
        output = F.softmax(output, dim=1)
    loss = criterion(output, target)
    return output, loss

def test_batch(model, image, target, criterion):
    output = model(image)
    if isinstance(criterion, nn.L1Loss):
        output = F.softmax(output, dim=1)
    loss = criterion(output, target)
    return output, loss