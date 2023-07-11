import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

# 用于对输入进行固定填充，返回填充后的输入。
def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1) # 计算有效的卷积核大小，即考虑了膨胀率的卷积核大小。
    pad_total = kernel_size_effective - 1 # 计算总的填充大小，即卷积核大小减去1。
    pad_beg = pad_total // 2    # 计算填充的开始位置和结束位置，即总的填充大小除以2。
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end)) # 使用F.pad函数对输入进行填充，填充的大小为(pad_beg, pad_end, pad_beg, pad_end)。
    return padded_inputs

# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()
        # 在初始化函数中，定义了三个卷积层：conv1是一个depthwise卷积层，bn是一个批归一化层，pointwise卷积层。
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    # (输入通道，输出通道，同种conv数量，步长，空洞率，是否在从relu开始，是否为最后（20）模块)
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        # 如果planes不等于inplanes或者stride不等于1，则需要进行跳跃连接，使用一个1x1的卷积层和批归一化层来调整维度。
        # 区别block1-3与block4-19
        # 满足条件为block1-3，定义1X1卷积和卷积后的bn层
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes) # batch_normal
        else:   # 不满足条件为block4-19，不定义skip
            self.skip = None
        # 定义relu层
        # inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
        self.relu = nn.ReLU(inplace=True)
        rep = []    # 空的列表rep，用于存储残差块的层。

        filters = inplanes  # 记录输入通道数
        # 根据grow_first参数的值，决定是先增加通道数还是先减少通道数
        # 为True，则先增加通道数，即先经过一个ReLU激活函数、一个可分离卷积层和一个批归一化层，然后将filters更新为planes
        # 为False，则先减少通道数，即先经过一个ReLU激活函数、一个可分离卷积层和一个批归一化层，然后将filters更新为inplanes。
        # 如果是每个组开始，则先定义一个由relu-sepconv-bn组成的可分离卷积块
        if grow_first:
            rep.append(self.relu)     # 第一层relu
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))  # 第二层3X3可分离卷积层
            rep.append(BatchNorm(planes))   #第三层bn层
            filters = planes  # filters更新为输出通道数

        # 定义rsp-1个由relu-sepconv-bn组成的可分离卷积块，每个残差块都包括一个ReLU激活函数、一个深度可分离卷积层和一个批归一化层。
        # 同上，第一层relu，第二层3X3可分离卷积层，第三层bn层
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        # 如果不是每个组开始，定义一个由relu-sepconv-bn组成的可分离卷积块
        # 同上，第一层relu，第二层3X3可分离卷积层，第三层bn层
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        # 如果stride等于1且is_last为True，则添加一个ReLU激活函数、一个可分离卷积层和一个批归一化层，用于最后的卷积操作
        # 判断是否是block1-3
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))  # 定义3X3下采样可分离卷积层
            rep.append(BatchNorm(planes))

        # 改写原来的stride=1为=2
        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
        # 如果start_with_relu为False，则将列表rep的第一个元素移除。
        # 是否保留块开始的RELU（）
        if not start_with_relu:
            rep = rep[1:]
        # 使用nn.Sequential将列表rep中的层组合成一个序列。
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        # 输入值放入rep中
        x = self.rep(inp)

        # 判断是否产生skip（逻辑结构为32-41行）
        # 如果产生，证明在block1-3中，这时候需要把skip和bn实例化出来
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, BatchNorm,
                 pretrained=True):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2 # block1-3中最下层步长
            middle_block_dilation = 1  # block4-19中空洞率
            exit_block_dilations = (1, 2)  # block20中空洞率
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        # # block1-3, inplanes = 3
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        # block4-19
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

        # Exit flow
        # block20
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x  # 四分之一大小
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)



if __name__ == "__main__":
    import torch
    model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
