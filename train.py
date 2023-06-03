import torch
from torch import nn, optim, tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
import time
from model import VGG16

# 参数
batch_size = 32
num_print = 100
epochs = 30
lr = 0.01
step_size = 10  # 没n次epoch更新一次学习率


# 数据获取，预处理
def data_preprocessing():
    """
    transforms.Compose(),将一系列的transforms有序组合,实现按照这些方法依次对图像操作
    ToTensor()使图片数据转换为tensor张量,这个过程包含了归一化,图像数据从0~255压缩到0~1,这个函数必须在Normalize之前使用
    实现原理,即针对不同类型进行处理,原理即各值除以255,最后通过torch.from_numpy将PIL Image或者
    numpy.ndarray()针对具体类型转成torch.tensor()数据类型
    #Normalize()是归一化过程,ToTensor()的作用是将图像数据转换为(0,1)之间的张量,Normalize()则使用公式(x-mean)/std
    :return:
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.226, 0.224, 0.225))
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.226, 0.224, 0.225))
    ])
    # root:cifar-10 的根目录,data_path
    # train:True=训练集, False=测试集
    # transform:(可调用,可选)-接收PIL图像并返回转换版本的函数
    # download:true=从互联网上下载数据,并将其放在root目录下,如果数据集已经下载,就什么都不干
    train_dataset = datasets.CIFAR10(root='/dataset/train', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='/dataset/test', train=False, transform=transform, download=True)

    return train_dataset, test_dataset


# 数据增强:随机翻转
train_dataset, test_dataset = data_preprocessing()

"""
batch_size:如果有50000张训练集,则相当于把训练集平均分成(50000/batch_size)份,每份batch_size张图片
train_loader中的每个元素相当于一个分组,一个组中batch_size图片,

shuffle:设置为True时会在每个epoch重新打乱数据(默认:False),一般在训练数据中会采用
num_workers:这个参数必须>=0,0的话表示数据导入在主进程中进行,其他大于0的数表示通过多个进程来导入数据,可以加快数据导入速度
drop_last:设定为True如果数据集大小不能被批量大小整除的时候,将丢到最后一个不完整的batch(默认为False)
"""

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VGG16().to(device)

loss_function = nn.CrossEntropyLoss()  # 用于多分类问题的损失函数

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001)

'''
scheduler 就是为了调整学习率设置的，我这里设置的gamma衰减率为0.5，step_size为10，也就是每10个epoch将学习率衰减至原来的0.5倍。

optimizer(Optimizer):要更改学习率的优化器
milestones(list):递增的list,存放要更新的lr的epoch
gamma:(float):更新lr的乘法因子
last_epoch:：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1
'''

schedule = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

# 训练
loss_list = []  # 为了后续画出损失图
start = time.time()
for epoch in range(epochs):

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = loss_function(outputs, labels).to(device)
        # 反向传播
        optimizer.zero_grad()  # 清空梯度，在每次应用新的梯度时,要把原来的梯度清零,否则梯度会累加
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        loss_list.append(loss.item())

        if (i + 1) % num_print == 0:
            print(f"[{epoch + 1} epoch, {i + 1}]  \t loss:{running_loss / num_print:.6f}")
            running_loss = 0.0
    lr_1 = optimizer.param_groups[0]['lr']
    print(f"learn_rate:{lr_1}")
    schedule.step()

end = time.time()
print(f"time cost:{end - start}")

# 测试
model.eval()
correct = 0.0
total = 0

with torch.no_grad():  # 测试集不需要反向传播
    print("===================test===================")
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        #  _,pred = torch.max(outputs,dim=1)# 取最大值所在的索引位置作为模型的预测结果
        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        total += inputs.size(0)  # 返回这个batch的样本数量，累加了每个batch中的样本数，最终得到整个数据集的样本总数
        # torch.eq() 比较两个张量是否相等
        # correct += torch.eq(pred, labels).sum().item()
        # 正确的预测数量
        correct += torch.sum(torch.eq(pred, labels)).item()

        print("Accucacy: {:.5f}".format(correct / total))
