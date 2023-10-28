# train.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def train(model, train_dataloader, criterion, optimizer, scheduler, device, class_names, train_set):
    model.train()
    torch.cuda.empty_cache()
    running_cls_loss = 0.0
    running_cls_corrects = 0
    start_time = time.time()
    all_y_true = []
    all_y_pred = []
    
    for batch_idx, (image, target) in enumerate(train_dataloader):
        image = image.to(device)
        target = target.to(device)
        one_hot_target = F.one_hot(target, len(class_names)).to(device)

        # train model
        if isinstance(criterion, nn.L1Loss):
            outputs, loss = train_batch(model, image, one_hot_target, criterion)
        else:
            outputs, loss = train_batch(model, image, target, criterion)

        _, preds = torch.max(outputs, 1)

        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')
        running_cls_loss += loss.item()
        running_cls_corrects += torch.sum(preds == target.data)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        all_y_true.extend(target.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())

    end_time = time.time()
    epoch_training_time = end_time - start_time

    epoch_loss = running_cls_loss / len(train_dataloader)
    epoch_acc = running_cls_corrects.double() / len(train_set)
    
    confusion = confusion_matrix(all_y_true, all_y_pred)
    report = classification_report(all_y_true, all_y_pred, output_dict=True)

#     print("Confusion Matrix:")
#     print(confusion)
#     print("Classification Report:")
#     print(report)
    
    return epoch_loss, epoch_acc.cpu().detach().numpy().tolist(), epoch_training_time, report, confusion
