import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary

# VGG16 model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # 1
            # stride:卷积核每次滑动步长
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 对64通道特征图进行Batch Normalization,批归一化
            nn.BatchNorm2d(num_features=64),  # num_features 表示输入张量的通道数，也就是卷积层的输出特征图数量。
            # 默认情况下，nn.ReLU()的输出会存储在一个新的张量中，而则表示将其计算结果直接覆盖输入张量，
            # 从而节省了额外的内存开销。这种原地操作可以提高模型的计算效率，适用于内存和计算能力所限的情况。
            nn.ReLU(inplace=True),

            # 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # 进行2x2的最大池化操作，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 6
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 7
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 8
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 9
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 11
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 12
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # 13
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier = nn.Sequential(
            # 14
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # 15
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 16
            nn.Linear(in_features=256,out_features=10)
        )

    def forward(self,x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(-1,512) # 因为classifier input_features = 512
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# net = VGG16()
# summary(net,(3,224,224))

