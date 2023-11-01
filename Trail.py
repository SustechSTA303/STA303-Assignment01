import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# random seed
SEED = 1 
NUM_CLASS = 10

# Training
BATCH_SIZE = 128
EVAL_INTERVAL=1

# Optimizer
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
STEP=5
GAMMA=0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


# Define your neural network model here
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

# Define a function to train the model
def train(model, loss_fn, optimizer,scheduler, train_loader, test_loader, num_epochs):
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.to('cpu').size(0)
            correct_train += predicted.to('cpu').eq(targets.to('cpu')).sum().item()

        # Calculate and store training accuracy and loss for this epoch
        train_accuracy = 100.0 * correct_train / total_train
        train_loss_history.append(total_train_loss / len(train_loader))
        train_acc_history.append(train_accuracy)

        scheduler.step()

        model.eval()
        total_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                outputs = model(inputs.to(device))
                loss = loss_fn(outputs, targets.to(device))
                total_test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.to('cpu').eq(targets.to('cpu')).sum().item()

        # Calculate and store testing accuracy and loss for this epoch
        test_accuracy = 100.0 * correct_test / total_test
        test_loss_history.append(total_test_loss / len(test_loader))
        test_acc_history.append(test_accuracy)

        print(f"Epoch [{epoch}/{num_epochs}] - Loss on Train: {train_loss_history[-1]:.4f} - Acc on Train: {train_accuracy:.2f}% - Loss on Test: {test_loss_history[-1]:.4f} - Acc on Test: {test_accuracy:.2f}")

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history

# def l1Loss_corr(output,target):
#     target_onehot=nn.functional.one_hot(target,num_classes=10)
#     cri=nn.L1Loss()
#     output=F.softmax(output,dim=1)
#     return cri(output,target_onehot)

# def l1Loss(output,target):
#     target_onehot=nn.functional.one_hot(target,num_classes=10)
#     cri=nn.L1Loss()
#     return cri(output,target_onehot)

class L1loss_corr(nn.Module):
    def __init__(self):
        super(L1loss_corr,self).__init__()

    def forward(self,logits,target):
        x_softmax=F.softmax(logits,dim=1)
        target_onehot=nn.functional.one_hot(target,num_classes=10)
        return F.l1_loss(x_softmax,target_onehot)


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, num_classes)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss =(1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

# Define the number of training trials
num_trials = 5
num_epochs = 30  # You can adjust the number of epochs as needed

# Create a dictionary to record the results
results = {}

# Loop through each trial with different loss functions
for trial in range(num_trials):
    model = ConvNet()  # Create your model
    model = nn.DataParallel(model,device_ids=[0,1])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)

    loss_fn=L1loss_corr()
    loss_name='l1loss + softmax'

    # if trial % 4 == 0:
    #     loss_fn = l1Loss
    #     loss_name = "L1loss"
    # elif trial % 4 == 1:
    #     loss_fn = nn.CrossEntropyLoss()
    #     loss_name = "CrossEntropy"
    # elif trial % 4 ==2:
    #     # Define the Focal Loss function
    #     loss_fn = MultiClassFocalLossWithAlpha(gamma=0.5)
    #     loss_name = "FocalLoss0.5"
    # else:
    #     loss_fn = MultiClassFocalLossWithAlpha(gamma=2)
    #     loss_name = "FocalLoss2"

    train_loss_history, train_acc_history, test_loss_history, test_acc_history = train(model, loss_fn, optimizer, scheduler,train_dataloader, test_dataloader, num_epochs)

    # Store the results in the dictionary
    if loss_name not in results:
        results[loss_name] = []
    results[loss_name].append((train_loss_history, train_acc_history, test_loss_history, test_acc_history))


    # Write the results to a text file
with open("results_sup.txt", "w") as file:
    for loss_name, trials in results.items():
        file.write(f"{loss_name} results:\n")
        for trial_num, (train_loss, train_acc, test_loss, test_acc) in enumerate(trials):
            for epoch, (train_loss_val, train_acc_val, test_loss_val, test_acc_val) in enumerate(zip(train_loss, train_acc, test_loss, test_acc)):
                file.write(f"Trial {trial_num + 1} Epoch {epoch + 1}: Loss on Train={train_loss_val:.4f}, Acc on Train={train_acc_val:.2f}%, Loss on Test={test_loss_val:.4f}, Acc on Test={test_acc_val:.2f}\n")
