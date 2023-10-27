# Assignment1 report
## 12110208宋子烨

### L1Loss
#### 代码部分

loss function直接使用nn自带的L1Loss
```python
criterion = nn.L1Loss()
```
根据L1Loss的公式：
$$
L1Loss =\frac{1}{n} * \sum_{i}|y_{pred_i} - y_{true_i}|
$$

其中：

- L1 Loss 表示整个数据集的平均绝对误差。
- n 表示样本的总数量。
- &Sigma; 表示求和符号
- y<sub>pred<sub>i</sub></sub> 表示的是第i个样本预测值
- y<sub>true<sub>i</sub></sub> 表示的是第i个样本真实值


由于传入的target是对应的class名，我们需要先对target使用one-hot编码

```python
target = F.one_hot(target,10)
```

再训练模型，下图展示为最后一组训练结果

![L1LossResult](C:\Users\32048\Desktop\STUDY\大三上\Artificial Intelligence\Assingment1\L1LossResult.png)

可以看出模型的拟合结果非常差，仅有10%。
#### 分析问题，总结

1. 由于L1Loss函数是用于测量连续值之间的差异的，并不适合用于测量分类问题的预测误差。

2. 由于L1Loss是不可微分的，因为它在零点处的导数是不确定的（可以是0或任何值，因为它在零点不连续）。这使得它不适合用于优化分类问题中的模型，本次分类问题需要使用随机梯度下降(SGD)优化算法来调整模型参数，而这个算法依赖于损失函数的可微性。


### CrossEntropyLoss
#### 代码部分
```python
criterion = nn.CrossEntropyLoss()
```
根据CrossEntropyLoss的公式：
$$
H(y,p) = -\sum_{i}(y_i*log(p_i))
$$

其中：

- y表示真实值
- p<sub>i</sub>是模型预测的类别概率，此处使用了softmax函数

这个公式中，不需要将目标转换成 one-hot 编码，因为 y 是一个整数，表示真实类别的索引。Cross-Entropy Loss 会自动考虑目标的离散性，并计算预测类别概率与真实类别的交叉熵。

训练模型，下图展示为最后一组训练结果
![CrossEntropyLossResult](C:\Users\32048\Desktop\STUDY\大三上\Artificial Intelligence\Assingment1\CrossEntropyLossResult.png)

#### 分析问题，总结

相较于L1Loss函数，CrossEntropyLoss函数大大提高了测试集的准确率，将近60%。可以认为CrossEntropyLoss函数对于处理多分类问题表现良好。

### Focal loss 
#### 代码部分
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        
        # transform into one-hot code
        target_one_hot = F.one_hot(target, num_classes=input.size(1))
        
        # compute focal loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt * target_one_hot
        
        # apply alpha 
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        return torch.mean(focal_loss)
```

其中：

- `FocalLoss` 类继承了 PyTorch 的 `nn.Module` 类，它是一个可自定义的 PyTorch 模块。

- `__init__` 方法是构造函数，接受两个参数：

  - `gamma` 是 Focal Loss 中的调节参数，控制难易分类样本的权重。通常，较小的 `gamma` 值会更关注难分类的样本。

  - `alpha` 是一个权重向量，用于处理类别不平衡问题。如果为 `None`，则不使用权重。

- `forward` 方法是前向传播函数，用于计算 Focal Loss。它接受两个参数：

  - `input` 是模型的预测输出，通常是一个概率分布，其中每个类别的概率由模型预测。
  - `target` 是真实的类别标签。

由于我们对类别之前并无权重要求，所以将alpha设置成none

##### gamma = 0.5

对于loss function，
```python
criterion = FocalLoss(0.5,None)
```

训练模型，下图展示为最后一组训练结果
![FocalLoss(gamma=0.5)](C:\Users\32048\Desktop\STUDY\大三上\Artificial Intelligence\Assingment1\FocalLoss(gamma=0.5).png)
##### gamma = 2

对于loss function，
```python
criterion = FocalLoss(2,None)
```

训练模型，下图展示为最后一组训练结果

![FocalLoss(gamma=2)](C:\Users\32048\Desktop\STUDY\大三上\Artificial Intelligence\Assingment1\FocalLoss(gamma=2).png)

#### 分析问题，总结

可以看出，不论是gamma=0.5或者gamma= 2，相较于CE loss，对于Train Loss及Test Loss都已极低 

但是准确率却并无显著提升。

猜想是：Focal Loss 主要用于解决类别不平衡问题，即某些类别的样本数量明显少于其他类别。如果你的数据集本身没有类别不平衡问题，或者已经采取其他方法来解决了这个问题，那么使用 Focal Loss 可能不会带来明显的改进。

### 其他提升
#### 更换激活函数
对于模型的激活函数，默认的是标准的relu函数，我们全部将其替换成nn.LeakyReLU(0.1)，并测试其在应用CE Loss上的效果，如下图所示![CELossWithLeakyReLU](C:\Users\32048\Desktop\STUDY\大三上\Artificial Intelligence\Assingment1\CELossWithLeakyReLU.png)

可以看出，相比于原本的CE Loss，应用LeakyReLU并无明显变化

#### 再增加一个卷积层
对于模型，我们敏锐的看出测试集的准确率不如训练集，推测是可能模型欠拟合，所以我们增加一个卷积层，增加神经网络的深度，代码如下：
```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)  # 添加一个额外的卷积层
        self.fc1 = nn.Linear(16 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))  # 在forward方法中使用新的卷积层
        x = x.view(-1, 16 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

```

测试其在应用CE Loss上的效果，如下图所示：

![addAnotherLayer](C:\Users\32048\Desktop\STUDY\大三上\Artificial Intelligence\Assingment1\addAnotherLayer.png)

可以看出，在增加一个卷积层后，模型拟合效果反而略微变差。在现阶段我无法作出解释😭😭😭