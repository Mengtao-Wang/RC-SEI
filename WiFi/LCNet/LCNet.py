import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LazyLinear
from thop import profile
from thop import clever_format
from torchsummary import summary
# from torchinfo import summary
import torchsnooper
from torchstat import stat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CReLU(nn.Module):
    """
    Complex-valued ReLU (CReLU)
    """

    def __init__(self, complex_axis=1, inplace=False):
        super(CReLU, self).__init__()
        self.r_relu = nn.ReLU(inplace=inplace)
        self.i_relu = nn.ReLU(inplace=inplace)
        self.complex_axis = complex_axis

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = self.r_relu(real)
        imag = self.i_relu(imag)
        return torch.cat([real, imag], self.complex_axis)


class ComplexConv1d(nn.Module):
    """
    Complex-valued Convolution (CC)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, complex_axis=1):

        super(ComplexConv1d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if groups == in_channels:
            self.groups = groups // 2
        else:
            self.groups = 1
        self.dilation = dilation
        self.bias = bias
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)
        self.imag_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        return torch.cat([real, imag], self.complex_axis)


# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55
class ComplexBatchNorm(torch.nn.Module):
    """
    Complex-valued Batch Normalization (CBN)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, complex_axis=1):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)

        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(self.num_features))
            self.register_buffer('RMi', torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, inputs):
        # self._check_input_dim(xr, xi)

        xr, xi = torch.chunk(inputs, 2, self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)

        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv1D, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class Channel_Shuffle(nn.Module):
    """
    input:  sample_length x channel
    output: channel x sample_length
    """

    def __init__(self, groups=2):
        super(Channel_Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, length = x.size()
        channels_per_group = num_channels // self.groups

        # Reshape
        x = x.view(batch_size, length, num_channels)
        x = x.view(batch_size, self.groups, channels_per_group, length)

        # Transpose
        x = x.transpose(1, 2).contiguous()

        # Flatten
        x = x.view(batch_size, -1, length)

        return x


# @torchsnooper.snoop()
class DSECA(nn.Module):
    """
    input: channel x sample_length
    output: channel x sample_length
    """

    def __init__(self, channel=32, b=1, gamma=2):
        super(DSECA, self).__init__()
        self.b = b
        self.gamma = gamma
        self.channel = channel
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        # 计算自适应卷积核大小
        # kernel_size = int(abs((math.log(int(self.channel), 2) + self.b) / self.gamma))
        # print(kernel_size )
        # if kernel_size % 2 == 0:
        #     kernel_size = kernel_size + 1  # 如果卷积核大小是偶数，就加一变奇数

        # Global Average Pooling
        x_GAP = self.global_avg_pool(x)
        x_GAP = x_GAP.view(-1, 1, self.channel)
        x_GAP = self.conv1(x_GAP)
        X_GAP = torch.sigmoid(x_GAP)

        # Global Max Pooling
        x_GMP = self.global_max_pool(x)
        x_GMP = x_GMP.view(-1, 1, self.channel)
        x_GMP = self.conv1(x_GMP)
        X_GMP = torch.sigmoid(x_GMP)

        # Concatenate and apply activation
        x_Mask = torch.cat([x_GAP, x_GMP], 2)
        x_Mask = torch.sigmoid(x_Mask)
        x_Mask = x_Mask.view(-1, self.channel, 1)
        x = x * x_Mask
        return x


# @torchsnooper.snoop()
class DWConvMobile(nn.Module):
    def __init__(self, n_neuron):
        super(DWConvMobile, self).__init__()
        self.conv = SeparableConv1D(n_neuron * 2, n_neuron * 2, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm1d(n_neuron * 2)
        self.relu = nn.ReLU()
        self.channel_shuffle = Channel_Shuffle(n_neuron * 2)
        self.attention = DSECA(channel=n_neuron * 2, b=1, gamma=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.channel_shuffle(x)
        x = self.attention(x)
        return x


class LCNet(nn.Module):
    def __init__(self, num_classes, n_neuron, n_mobileunit):
        """ Initialize a ULNN
        Args: num_classes (int): the number of classes
        """
        super(LCNet, self).__init__()
        self.n_mobileunit = n_mobileunit
        self.conv1 = ComplexConv1d(2, n_neuron * 2, 9, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = ComplexBatchNorm(n_neuron * 2)

        self.mobile_units = nn.ModuleList([DWConvMobile(n_neuron) for _ in range(n_mobileunit)])
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        for i, mobile_unit in enumerate(self.mobile_units):
            x = mobile_unit(x)
        x = x.view(-1, 1536)
        output = self.fc1(x)

        return output

@torchsnooper.snoop()
class base_LCNet(nn.Module):
    def __init__(self, num_classes, n_neuron, n_mobileunit):
        """ Initialize a ULNN
        Args: num_classes (int): the number of classes
        """
        super(base_LCNet, self).__init__()
        self.n_mobileunit = n_mobileunit
        self.conv1 = ComplexConv1d(2, n_neuron * 2, 5, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = ComplexBatchNorm(n_neuron * 2)

        self.mobile_units = nn.ModuleList([DWConvMobile(n_neuron) for _ in range(n_mobileunit)])
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(1536)))
        self.linear = LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        for i, mobile_unit in enumerate(self.mobile_units):
            x = mobile_unit(x)
        x = self.flatten(x)
        lamda = self.lamda

        s_f = int(1536 - torch.sum(lamda == 0))
        print('稀疏特征数量 = ', s_f)

        x *= lamda
        embedding = F.relu(x)
        output = self.linear(embedding)
        return s_f, output

class prune_LCNet(nn.Module):
    def __init__(self, num_classes, n_neuron, n_mobileunit, f_list, m):
        """ Initialize a ULNN
        Args: num_classes (int): the number of classes
        """
        super(prune_LCNet, self).__init__()
        self.f_list = f_list
        self.n_mobileunit = n_mobileunit
        self.conv1 = ComplexConv1d(2                                                                                                                                                                                                                                  , n_neuron * 2, 5, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = ComplexBatchNorm(n_neuron * 2)

        self.mobile_units = nn.ModuleList([DWConvMobile(n_neuron) for _ in range(n_mobileunit)])
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.lamda = torch.nn.Parameter(torch.Tensor(np.ones(m)))  # prune模型不再需要这层结构，不参与前向传播
        self.linear = LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        for i, mobile_unit in enumerate(self.mobile_units):
            x = mobile_unit(x)

        x = self.flatten(x)
        x = x[:, self.f_list]
        embedding = F.relu(x)
        output = self.linear(embedding)
        return output


if __name__ == '__main__':
    # depth  flatten   parameters   flatten        acc
    #  *7*     1536      57804     24592(43%)     99.90%
    #   8      768       50192     12304(24%)     99.80%
    model_1 = base_LCNet(num_classes=16, n_neuron=32, n_mobileunit=7)
    summary(model_1, input_size=(2, 6000), batch_size=1, device="cpu")
