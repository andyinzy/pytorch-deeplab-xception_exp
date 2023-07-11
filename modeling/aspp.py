import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
"""
这是_ASPPModule类的定义，它是ASPP模块的一个子模块。
ASPP模块通过多个不同尺度的空洞卷积（atrous convolution）来捕捉不同尺度的上下文信息，以提高语义分割的性能。
具体来说，该类的初始化函数接受输入通道数（inplanes）、输出通道数（planes）、卷积核大小（kernel_size）、填充（padding）、扩张率（dilation）和批归一化函数（BatchNorm）作为参数。
在初始化函数中，它定义了一个空洞卷积层（atrous_conv），一个批归一化层（bn）和一个ReLU激活函数（relu）。
在前向传播函数中，输入通过空洞卷积层、批归一化层和ReLU激活函数，然后返回激活后的输出。
初始化权重函数（_init_weight）用于初始化模块中的卷积层和批归一化层的权重。

"""



class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

""""
这是ASPP类的定义，它是整个ASPP模块的主体。在初始化函数中，根据所选择的backbone和output_stride，确定输入通道数（inplanes）和空洞卷积的扩张率（dilations）。
然后，它定义了四个_ASPPModule子模块（aspp1、aspp2、aspp3和aspp4），一个全局平均池化层（global_avg_pool），一个卷积层（conv1），一个批归一化层（bn1），一个ReLU激活函数（relu）和一个dropout层（dropout）。
在前向传播函数中，输入通过四个_ASPPModule子模块和全局平均池化层，然后通过插值操作将全局平均池化层的输出与其他四个子模块的输出进行拼接。最后，拼接后的特征图通过卷积层、批归一化层、ReLU激活函数和dropout层，最终输出。
初始化权重函数（_init_weight）用于初始化模块中的卷积层和批归一化层的权重。
总的来说，这段代码定义了一个ASPP模块，用于图像语义分割任务。ASPP模块通过多个不同尺度的空洞卷积和全局平均池化来捕捉不同尺度的上下文信息，并通过拼接和卷积操作来融合这些信息，以提高语义分割的性能。
"""

def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)