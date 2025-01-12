import torch
import torch.nn as nn
from torchstat import stat
from torchinfo import summary
#from torchsummary import summary

import torchsnooper
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear
import torch.nn.functional as F
import numpy as np
from thop import profile
from thop import clever_format

# @torchsnooper.snoop()
class MCLDNN(nn.Module):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone
    The input for LSTM is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=16):
        super(MCLDNN, self).__init__()
        self.num_classes = num_classes
        """
        原MCLDNN网络是处理RML2016.10a数据调制信号, 信号长度为128, 对于ADS-B数据来说欠拟合
        增加两层小卷积核卷积层,提高特征提取能力
        """
        self.conv0 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(num_features=64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 50, kernel_size=(2, 8), padding='same', ),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(64, 50, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(64, 50, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(50)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        self.lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, num_layers=2)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 256),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 512),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(512, 16),
            )

    def forward(self, x):
        x  = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x[:, :, 0, :])
        x3 = self.conv3(x[:, :, 1, :])
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x = torch.reshape(x5, [-1, x5.shape[3], 100])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])

        return (x,)

class base_MCLDNN(nn.Module):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone
    The input for LSTM is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=16):
        super(base_MCLDNN, self).__init__()
        self.num_classes = num_classes
        """
        原MCLDNN网络是处理RML2016.10a数据调制信号, 信号长度为128, 对于ADS-B数据来说欠拟合
        增加两层小卷积核卷积层,提高特征提取能力
        """
        self.conv0 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(num_features=64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 50, kernel_size=(2, 8), padding='same', ),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(64, 50, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(64, 50, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(50)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        self.lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, num_layers=2)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 256),
                nn.SELU(),
                nn.Linear(256, 512),
                nn.SELU()

            )

        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(512)))
        self.linear = LazyLinear(num_classes)


    def forward(self, x):
        x  = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x[:, :, 0, :])
        x3 = self.conv3(x[:, :, 1, :])
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x = torch.reshape(x5, [-1, x5.shape[3], 100])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])
        lamda = self.lamda
        # print('特征优化数量 =', torch.sum(lamda == 0))
        x *= lamda
        embedding = F.relu(x)
        output = self.linear(embedding)
        return output

class prune_MCLDNN(nn.Module):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone
    The input for LSTM is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            TUe default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=16, f_list=None, m=None):
        super(prune_MCLDNN, self).__init__()
        self.num_classes = num_classes
        self.f_list = f_list
        """
        原MCLDNN网络是处理RML2016.10a数据调制信号, 信号长度为128, 对于ADS-B数据来说欠拟合
        增加两层小卷积核卷积层,提高特征提取能力
        """
        self.conv0 = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ZeroPad2d(padding=(1, 1, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(num_features=64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 50, kernel_size=(2, 8), padding='same', ),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(64, 50, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(64, 50, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(50)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        self.lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, num_layers=2)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 256),
                nn.SELU(),
                nn.Linear(256, 512),
                nn.SELU()
            )
        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(m)))
        self.linear = LazyLinear(num_classes)


    def forward(self, x):
        x  = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x[:, :, 0, :])
        x3 = self.conv3(x[:, :, 1, :])
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x = torch.reshape(x5, [-1, x5.shape[3], 100])
        x, _ = self.lstm(x)
        if self.num_classes > 0:
            x = self.classifier(x[:, -1, :])
        x = x[:, self.f_list]
        embedding = F.relu(x)
        output = self.linear(embedding)
        return output

if __name__ == "__main__":

    model_1 = base_MCLDNN(num_classes=16)
    summary(model_1, input_size=(1, 2, 6000), batch_dim=1, device="cpu")


