
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation, stride=stride, padding=dilation, bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.gn1 = nn.GroupNorm(int(planes/2),planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.gn2 = nn.GroupNorm(int(planes/2),planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(int(planes/2),planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.gn2 = nn.GroupNorm(int(planes/2),planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(int(planes*4/2),planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_seg_classes, shortcut_type='B', no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(2, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.gn1 = nn.GroupNorm(int(64/2),64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[1], shortcut_type, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer3 = self._make_layer(block, 256, layers[3], shortcut_type, stride=1, dilation=4)

        self.conv_seg = nn.Sequential(
            nn.Conv3d(256 * block.expansion, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(int(32/2),32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(int(32/2),32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_seg_classes, kernel_size=1,  stride=(1, 1, 1), bias=False),
        )

        self.conv_seg_up = nn.Sequential(
            nn.Conv3d(num_seg_classes, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(int(256 / 2), 256),
        )
        self.relu_out = nn.ReLU(inplace=True)
        self.linear_reg = nn.Linear(256, 5)
        self.linear_cls = nn.Linear(256 * block.expansion, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):  # 仿造 Tingsong_Yu/Pytorch_tutorial main.py Line 90
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride, no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.GroupNorm(int(planes * block.expansion/2),planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward_once(self, x, y):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x_seg_feature = self.conv_seg(self.relu(x))
        x_seg = torch.sigmoid(x_seg_feature)
        x_encoding = self.relu_out(x + 0)
        x_encoding = F.adaptive_avg_pool3d(x_encoding, (1,1,1))
        x_encoding = x_encoding.view(x_encoding.size(0), -1)
        x_reg = self.linear_reg(x_encoding)
        softmax_func = nn.Softmax(dim=1)
        x_output_reg = softmax_func(x_reg)
        x_feature1 = self.linear_cls(x_encoding)
        x_output = x_feature1

        y = self.conv1(y)
        y = self.gn1(y)
        y = self.relu(y)
        y = self.layer1(y)
        y = self.relu(y)
        y = self.layer2(y)
        y = self.relu(y)
        y = self.layer3(y)
        y_seg_feature = self.conv_seg(self.relu(y))
        y_seg = torch.sigmoid(y_seg_feature)
        y_encoding = self.relu_out(y + 0)
        y_encoding = F.adaptive_avg_pool3d(y_encoding, (1,1,1))
        y_encoding = y_encoding.view(y_encoding.size(0), -1)
        y_output = self.linear_reg(y_encoding)

        return x_output, x_output_reg, x_seg, y_output, y_seg


    def forward(self, sch_0):
        x_output, x_output_reg, x_seg, y_output, y_seg = self.forward_once(sch_0, sch_0)
        return x_output

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


