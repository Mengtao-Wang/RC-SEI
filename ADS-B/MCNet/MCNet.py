import torch
import torch.nn as nn
from torchstat import stat
from thop import profile
from thop import clever_format
from torchsummary import summary
import torchsnooper
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear
import torch.nn.functional as F
import numpy as np


# @torchsnooper.snoop()
class _Pre_Block(nn.Module):
    def __init__(self):
        super(_Pre_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)

        return x

# @torchsnooper.snoop()
class _M_BlockA(nn.Module):
    def __init__(self, ):
        super(_M_BlockA, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(64, 32, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(3, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        sx = self.skip(x)
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = sx + x

        return x

# @torchsnooper.snoop()
class _M_BlockB(nn.Module):
    def __init__(self, has_pooling=False):
        super(_M_BlockB, self).__init__()
        self.has_pooling = has_pooling
        self.conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        if has_pooling:
            self.skip = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 0, 1, 0)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(32, 48, kernel_size=(3, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
            )
            self.conv2 = nn.Sequential(
                nn.ZeroPad2d(padding=(1, 1, 0, 0)),
                nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(48)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
            )
        else:
            self.skip = nn.Identity()
            self.conv1 = nn.Sequential(
                nn.Conv2d(32, 48, kernel_size=(3, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(48)
            )
            self.conv2 = nn.Sequential(
                nn.ZeroPad2d(padding=(1, 1, 0, 0)),
                nn.Conv2d(32, 48, kernel_size=(1, 3)),
                nn.ReLU(),
                nn.BatchNorm2d(48)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(32)
            )

    def forward(self, x):
        if x.shape[3] % 2 != 0:
            # 创建一个新的张量，其形状与 x 相同，但第四个维度加 1
            new_shape = list(x.shape)
            new_shape[3] += 1
            new_x = torch.zeros(new_shape, dtype=x.dtype, device=x.device)

            # 将原始张量 x 复制到新张量的前四个维度
            new_x[:, :, :, :-1] = x
        else:
            new_x = x
        sx = self.skip(new_x)
        x = self.conv(new_x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = sx + x

        return x

# @torchsnooper.snoop()
class _M_BlockC(nn.Module):
    def __init__(self):
        super(_M_BlockC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.skip = nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

    def forward(self, x):
        sx = self.skip(x)
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([sx, x1, x2, x3], dim=1)

        return x

#@torchsnooper.snoop()
class MCNet(nn.Module):
    """`MCNet <https://ieeexplore.ieee.org/abstract/document/8963964>`_ backbone
    The input for MCNet is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """


    def __init__(self, frame_length=4800, num_classes=10):
        """
        原MCNet网络处理信号长度为1024,ADS-B数据集信号长度为4800,原MCNet网络在该数据集上欠拟合。
        因此在原MCNet网络前添加两层小卷积核卷积层，将数据维度从4800降到1200。
        """
        super(MCNet, self).__init__()
        self.frame_length = frame_length # 信号长度
        self.num_classes = num_classes # 种类

        self.pad1 = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 2))
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)

        self.pad2 = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 2))
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)


        self.fea = nn.Sequential(
            nn.ZeroPad2d(padding=(3, 3, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            _Pre_Block(),
            _M_BlockA(),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockC(),
        )

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=(1, frame_length // 128)),
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(1024, num_classes),
            )

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)

        x = self.fea(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)

class base_MCNet(nn.Module):
    """`MCNet <https://ieeexplore.ieee.org/abstract/document/8963964>`_ backbone
    The input for MCNet is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """


    def __init__(self, frame_length=1200, num_classes=10):
        """
        原MCNet网络处理信号长度为1024,ADS-B数据集信号长度为4800,原MCNet网络在该数据集上欠拟合。
        因此在原MCNet网络前添加两层小卷积核卷积层，将数据维度从4800降到1200。
        """
        super(base_MCNet, self).__init__()
        self.frame_length = frame_length # 信号长度
        self.num_classes = num_classes # 种类

        # 1. 卷积层 x 2 input: 4800 → 1200
        self.pad1 = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 2))
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)

        self.pad2 = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 2))
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        # 2. 核心架构 input: 1200 → 1024
        self.fea = nn.Sequential(
            nn.ZeroPad2d(padding=(3, 3, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            _Pre_Block(),
            _M_BlockA(),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockC(),
        )

        # 3. 分类器 input 1024 → 10
        self.pool = nn.AvgPool2d(kernel_size=(2, frame_length // 128))
        self.Dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(512)))
        self.linear = LazyLinear(num_classes)


    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)

        x = self.fea(x)

        x = self.pool(x)
        lamda = self.lamda
        # print('特征优化数量 =', torch.sum(lamda == 0))
        # x = self.Dropout(x)
        x = self.flatten(x)
        x *= lamda
        embedding = F.relu(x)
        output = self.linear(embedding)
        return output

class prune_MCNet(nn.Module):
    """`MCNet <https://ieeexplore.ieee.org/abstract/document/8963964>`_ backbone
    The input for MCNet is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=1200, num_classes=10, f_list=None, m=None):
        """
        原MCNet网络处理信号长度为1024,ADS-B数据集信号长度为4800,原MCNet网络在该数据集上欠拟合。
        因此在原MCNet网络前添加两层小卷积核卷积层，将数据维度从4800降到1200。
        """
        super(prune_MCNet, self).__init__()
        self.frame_length = frame_length  # 信号长度
        self.num_classes = num_classes  # 种类
        self.f_list = f_list  # 稀疏特征索引

        # 1. 卷积层 x 2 input: 4800 → 1200
        self.pad1 = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 2))
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)

        self.pad2 = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 2))
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        # 2. 核心架构 input: 1200 → 1024
        self.fea = nn.Sequential(
            nn.ZeroPad2d(padding=(3, 3, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            _Pre_Block(),
            _M_BlockA(),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockB(),
            _M_BlockB(has_pooling=True),
            _M_BlockC(),
        )

        # 3. 分类器 input 1024 → 10
        self.pool = nn.AvgPool2d(kernel_size=(2, frame_length // 128))
        self.Dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(m)))
        self.linear = LazyLinear(num_classes)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.fea(x)
        x = self.pool(x)
        x = self.Dropout(x)

        x = self.flatten(x)
        x = x[:, self.f_list]
        embedding = F.relu(x)
        output = self.linear(embedding)
        return output


if __name__ == "__main__":

    model_1 = base_MCNet(frame_length=1200, num_classes=10)
    summary(model_1, input_size=(1, 2, 4800), batch_dim=1, device="cpu")
