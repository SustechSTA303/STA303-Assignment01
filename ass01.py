import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
import Focal
import time
import torchvision.models as models
import torch.nn.functional as F

epoch = 40
learning_rate = 0.01
dataset = "CIFAR10"  #CIFAR10,Caltech101
loss_function="MAE"  #MAE,CE,Focal0.5,Focal2

print('dataset:',dataset,'loss_function:',loss_function)

# 打开一个文件用于写入（如果文件不存在将会创建它，如果存在将会覆盖它）
file_path = "/data/lab/ass01/log.txt"  # 文件路径
with open(file_path, "a") as file:
    # 写入文本内容
    file.write('dataset: {} loss_function: {} learning_rate: {}\n'.format(dataset, loss_function,learning_rate))

if(torch.cuda.is_available()):
    print("use gpu")

# 定义数据转换


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)


# 加载数据集
if(dataset=="CIFAR10"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)), ])
    train_dataset = torchvision.datasets.CIFAR10(root='/data/lab/ass01/dataset',train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='/data/lab/ass01/dataset',train=False, download=False, transform=transform)
elif(dataset=="Caltech101"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)), ])


# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2,drop_last = True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=128,shuffle=True,num_workers=2,drop_last = True)

# 创建模型
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)  # 替换为输出10个类别的最终全连接层。
model = model.to(device)

'''model.backbone = nn.Sequential(*list(model.children())[:-2])
model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(1000, 10)  # 修改num_classes以匹配你的任务
)
'''

# 定义损失函数和优化器
if(loss_function=="MAE"):
    criterion = nn.L1Loss()
elif(loss_function=="CE"):
    criterion = nn.CrossEntropyLoss()
elif(loss_function=="Focal0.5"):
    criterion = Focal.FocalLoss(alpha=None, gamma=0.5)
elif(loss_function=="Focal2"):
    criterion = Focal.FocalLoss(alpha=None, gamma=2)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


def eval(loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for img, label in loader:
            if(loss_function=="MAE"):
                one_hot_labels = F.one_hot(labels, num_classes=10).to(torch.float32)
            img, label = img.to(device), label.to(device)
            logits = model(img)

            if(loss_function=="MAE"):
                test_loss += criterion(logits,one_hot_labels).item()
            else:
                test_loss += criterion(logits, label).item()
                
            _, predicted = logits.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            total_loss+=test_loss

    total_loss /= total
    test_acc = 100. * correct / total
    return test_loss, test_acc

t_start = time.time()

try:
# 训练模型
    for e in range(epoch):
        model.train()

        for batch in tqdm(train_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.zero_grad()
            outputs = model(inputs)
            if(loss_function=="MAE"):
                one_hot_labels = F.one_hot(labels, num_classes=10).to(torch.float32)
                loss = criterion(outputs, one_hot_labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        test_loss,test_acc = eval(test_loader)
        train_loss, train_acc = eval(train_loader)

        output_str = 'epoch: {} train acc: {} test acc: {} train loss: {} test loss: {} total time: {}\n'.format(e+1, train_acc,test_acc, train_loss,test_loss,time.time() - t_start)
        print(output_str)
        with open(file_path, "a") as file:
            # 写入文本内容
            file.write(output_str)

except KeyboardInterrupt:
    pass

save_path = '/data/lab/ass01/module_save'
torch.save(model.state_dict(), save_path)
print('finish training!')


