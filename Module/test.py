import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import test_batch
from utility import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def test(model, test_dataloader, criterion, device, class_names, test_set):
    print('Begin test......')
    model.eval()

    val_loss = 0.0
    val_corrects = 0
    all_y_true = []
    all_y_pred = []
    start_time2 = time.time()
    
    for batch_idx, (image, target) in enumerate(test_dataloader):
        image = image.to(device)
        target = target.to(device)
        one_hot_target = F.one_hot(target, len(class_names)).to(device)

        # test model
        if isinstance(criterion, nn.L1Loss):
            outputs, loss = test_batch(model, image, one_hot_target, criterion)
        else:
            outputs, loss = test_batch(model, image, target, criterion)

        _, preds = torch.max(outputs, 1)

        val_loss += loss.item()
        val_corrects += torch.sum(preds == target.data)
        all_y_true.extend(target.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())

    end_time2 = time.time()  # 记录结束时间
    epoch_testing_time = end_time2 - start_time2
    val_loss = val_loss / len(test_dataloader)
    val_corrects = val_corrects.double() / len(test_set)
    confusion = confusion_matrix(all_y_true, all_y_pred)
    report = classification_report(all_y_true, all_y_pred, output_dict=True)
#     print("Classification Report:")
#     print(report)
    
    return val_loss, val_corrects.cpu().detach().numpy().tolist(), epoch_testing_time, report, confusion
