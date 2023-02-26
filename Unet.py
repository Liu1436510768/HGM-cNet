from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dropblock import DropBlock3D
from model.U_Net2 import UNet2
from torchvision import datasets, transforms

# 残差注意力模块1
class Resat1(nn.Module):
    def __init__(self):
        super(Resat1, self).__init__()

        self.resat1 = ResBlock(96, 96)

        # 主干分支
        self.resat2 = ResBlock(96, 96)
        self.resat3 = ResBlock(96, 96)

        # 掩膜分支
        self.down1 = nn.MaxPool3d(2, stride=2)
        self.resat4 = ResBlock(96, 96)

        self.resat5 = ResBlock(96, 96)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')


        self.conv1 = nn.Conv3d(96, 96, 1, stride=1)
        self.conv2 = nn.Conv3d(96, 96, 1, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.resat6 = ResBlock(96, 96)

    def forward(self, input):
        rt1 = self.resat1(input)

        # 主干分支
        rt2 = self.resat2(rt1)
        rt3 = self.resat3(rt2)

        # 掩膜分支
        rtd1 = self.down1(rt1)
        rt4 = self.resat4(rtd1)

        rt5 = self.resat5(rt4)
        rtu1 = self.upsample1(rt5)

        conv1 = self.conv1(rtu1)
        conv2 = self.conv2(conv1)
        sig = self.sigmoid(conv2)

        rt = rt3 * sig
        rt = rt + rt3
        rt = self.resat6(rt)

        return rt


# 残差注意力模块2
class Resat2(nn.Module):
    def __init__(self):
        super(Resat2, self).__init__()

        self.resat1 = ResBlock(64, 64)

        # 主干分支
        self.resat2 = ResBlock(64, 64)
        self.resat3 = ResBlock(64, 64)

        # 掩膜分支
        self.down1 = nn.MaxPool3d(2, stride=2)
        self.resat4 = ResBlock(64, 64)
        self.down2 = nn.MaxPool3d(2, stride=2)
        self.resat5 = ResBlock(64, 64)

        self.resat6 = ResBlock(64, 64)

        self.resat7 = ResBlock(64, 64)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.resat8 = ResBlock(128, 64)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv3d(64, 64, 1, stride=1)
        self.conv2 = nn.Conv3d(64, 64, 1, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.resat9 = ResBlock(64, 64)

    def forward(self, input):
        rt1 = self.resat1(input)

        # 主干分支
        rt2 = self.resat2(rt1)
        rt3 = self.resat3(rt2)

        # 掩膜分支
        rtd1 = self.down1(rt1)
        rt4 = self.resat4(rtd1)
        rtd2 = self.down2(rt4)
        rt5 = self.resat5(rtd2)

        rt6 = self.resat6(rt4)

        rt7 = self.resat7(rt5)
        rtu1 = self.upsample1(rt7)
        c1 = torch.cat((rtu1, rt6), dim=1)
        rt8 = self.resat8(c1)
        rtu2 = self.upsample2(rt8)

        conv1 = self.conv1(rtu2)
        conv2 = self.conv2(conv1)
        sig = self.sigmoid(conv2)

        rt = rt3 * sig
        rt = rt + rt3
        rt = self.resat9(rt)

        return rt


# 残差注意力模块3
class Resat3(nn.Module):
    def __init__(self):
        super(Resat3, self).__init__()

        self.resat1 = ResBlock(48, 48)

        # 主干分支
        self.resat2 = ResBlock(48, 48)
        self.resat3 = ResBlock(48, 48)

        # 掩膜分支
        self.down1 = nn.MaxPool3d(2, stride=2)
        self.resat4 = ResBlock(48, 48)
        self.down2 = nn.MaxPool3d(2, stride=2)
        self.resat5 = ResBlock(48, 48)
        self.down3 = nn.MaxPool3d(2, stride=2)
        self.resat6 = ResBlock(48, 48)

        self.resat7 = ResBlock(48, 48)
        self.resat8 = ResBlock(48, 48)

        self.resat9 = ResBlock(48, 48)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.resat10 = ResBlock(96, 48)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.resat11 = ResBlock(96, 48)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv3d(48, 48, 1, stride=1)
        self.conv2 = nn.Conv3d(48, 48, 1, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.resat12 = ResBlock(48, 48)

    def forward(self, input):
        rt1 = self.resat1(input)

        # 主干分支
        rt2 = self.resat2(rt1)
        rt3 = self.resat3(rt2)

        # 掩膜分支
        rtd1 = self.down1(rt1)
        rt4 = self.resat4(rtd1)
        rtd2 = self.down2(rt4)
        rt5 = self.resat5(rtd2)
        rtd3 = self.down3(rt5)
        rt6 = self.resat6(rtd3)

        rt7 = self.resat7(rt5)
        rt8 = self.resat8(rt4)

        rt9 = self.resat9(rt6)
        rtu1 = self.upsample1(rt9)
        c1 = torch.cat((rtu1, rt7), dim=1)
        rt10 = self.resat10(c1)
        rtu2 = self.upsample2(rt10)
        c2 = torch.cat((rtu2, rt8), dim=1)
        rt11 = self.resat11(c2)
        rtu3 = self.upsample3(rt11)

        conv1 = self.conv1(rtu3)
        conv2 = self.conv2(conv1)
        sig = self.sigmoid(conv2)

        rt = rt3 * sig
        rt = rt + rt3
        rt = self.resat12(rt)

        return rt


# 残差注意力模块4
class Resat4(nn.Module):
    def __init__(self):
        super(Resat4, self).__init__()

        self.resat1 = ResBlock(32, 32)

        # 主干分支
        self.resat2 = ResBlock(32, 32)
        self.resat3 = ResBlock(32, 32)

        # 掩膜分支
        self.down1 = nn.MaxPool3d(2, stride=2)
        self.resat4 = ResBlock(32, 32)
        self.down2 = nn.MaxPool3d(2, stride=2)
        self.resat5 = ResBlock(32, 32)
        self.down3 = nn.MaxPool3d(2, stride=2)
        self.resat6 = ResBlock(32, 32)
        self.down4 = nn.MaxPool3d(2, stride=2)
        self.resat7 = ResBlock(32, 32)

        self.resat8 = ResBlock(32, 32)
        self.resat9 = ResBlock(32, 32)
        self.resat10 = ResBlock(32, 32)

        self.resat11 = ResBlock(32, 32)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.resat12 = ResBlock(64, 32)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.resat13 = ResBlock(64, 32)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.resat14 = ResBlock(64, 32)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv3d(32, 32, 1, stride=1)
        self.conv2 = nn.Conv3d(32, 32, 1, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.resat15 = ResBlock(32, 32)

    def forward(self, input):
        rt1 = self.resat1(input)

        # 主干分支
        rt2 = self.resat2(rt1)
        rt3 = self.resat3(rt2)

        # 掩膜分支
        rtd1 = self.down1(rt1)
        rt4 = self.resat4(rtd1)
        rtd2 = self.down2(rt4)
        rt5 = self.resat5(rtd2)
        rtd3 = self.down3(rt5)
        rt6 = self.resat6(rtd3)
        rtd4 = self.down4(rt6)
        rt7 = self.resat7(rtd4)

        rt8 = self.resat8(rt6)
        rt9 = self.resat9(rt5)
        rt10 = self.resat10(rt4)

        rt11 = self.resat11(rt7)
        rtu1 = self.upsample1(rt11)
        c1 = torch.cat((rtu1, rt8), dim=1)
        rt12 = self.resat12(c1)
        rtu2 = self.upsample2(rt12)
        c2 = torch.cat((rtu2, rt9), dim=1)
        rt13 = self.resat13(c2)
        rtu3 = self.upsample3(rt13)
        c3 = torch.cat((rtu3, rt10), dim=1)
        rt14 = self.resat14(c3)
        rtu4 = self.upsample4(rt14)

        conv1 = self.conv1(rtu4)
        conv2 = self.conv2(conv1)
        sig = self.sigmoid(conv2)

        rt = rt3 * sig
        rt = rt + rt3
        rt = self.resat15(rt)

        return rt


class Autofocus(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus, self).__init__()
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2

        self.conv1 = nn.Conv3d(in_channels, out_channels1, kernel_size=kernel, dilation=2, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels1)

        self.conv2 = nn.Conv3d(out_channels1, out_channels2, kernel_size=kernel, dilation=self.dilation_list[0], padding=2)
        self.convatt1 = nn.Conv3d(out_channels1, int(out_channels1 / 2), kernel_size=kernel, padding=1)
        self.convatt2 = nn.Conv3d(int(out_channels1 / 2), self.num_branches, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(in_channels, out_channels2, kernel_size=1), nn.BatchNorm3d(out_channels2))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # compute attention weights for the second layer
        feature = x.detach()    # 添加detach()时的requires_grad为False，因为对fea进行更改，所以并不会影响backward()。
        att = self.relu(self.convatt1(feature))
        att = self.convatt2(att)
        # att = torch.sigmoid(att)
        att = F.softmax(att, dim=1)
        # att = att[:, :, 1:-1, 1:-1, 1:-1]

        # linear combination of different dilation rates
        x1 = self.conv2(x)
        shape = x1.size()
        x1 = self.bn_list2[0](x1) * att[:, 0:1, :, :, :].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list2[i](x2)
            x1 += x2 * att[:, i:(i + 1), :, :, :].expand(shape)

        # if self.downsample is not None:
        # residual = self.downsample(residual)

        x = x1 + residual
        x = self.relu(x)
        return x


class Autofocus2(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus2, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches

        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=self.dilation_list[0], padding=2)
        self.convatt11 = nn.Conv3d(inplanes1, int(inplanes1 / 2), kernel_size=kernel, padding=1)
        self.convatt12 = nn.Conv3d(int(inplanes1 / 2), self.num_branches, kernel_size=1)
        self.bn_list1 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list1.append(nn.BatchNorm3d(outplanes1))

        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=self.dilation_list[0],padding=2)
        self.convatt21 = nn.Conv3d(outplanes1, int(outplanes1 / 2), kernel_size=kernel, padding=1)
        self.convatt22 = nn.Conv3d(int(outplanes1 / 2), self.num_branches, kernel_size=1)
        self.bn_list2 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))

        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        # compute attention weights in the first autofocus convolutional layer
        feature = x.detach()
        att = self.relu(self.convatt11(feature))
        att = self.convatt12(att)
        # att = torch.sigmoid(att)
        att = F.softmax(att, dim=1)
        # att = att[:, :, 1:-1, 1:-1, 1:-1]

        # linear combination of different dilation rates
        x1 = self.conv1(x)
        shape = x1.size()
        x1 = self.bn_list1[0](x1) * att[:, 0:1, :, :, :].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv1.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list1[i](x2)
            x1 += x2 * att[:, i:(i + 1), :, :, :].expand(shape)

        #if self.downsample is not None:
            # residual = self.downsample(residual)
        x = self.relu(x1)

        # compute attention weights for the second layer
        feature2 = x.detach()
        att2 = self.relu(self.convatt21(feature2))
        att2 = self.convatt22(att2)
        # att = torch.sigmoid(att)
        att2 = F.softmax(att2, dim=1)
        # att = att[:, :, 1:-1, 1:-1, 1:-1]

        # linear combination of different dilation rates
        x21 = self.conv2(x)
        shape = x21.size()
        x21 = self.bn_list2[0](x21) * att[:, 0:1, :, :, :].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x22 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x22 = self.bn_list2[i](x22)
            x21 += x22 * att2[:, i:(i + 1), :, :, :].expand(shape)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x21 + residual
        x = self.relu(x)
        return x


class Autofocus_single(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus_single, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.con = nn.Conv3d(inplanes1, outplanes2, kernel_size=1)

        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=2, padding=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)

        self.bn_list2 = nn.ModuleList()
        for i in range(len(self.padding_list)):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))

        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=self.dilation_list[0],padding=2)
        self.convatt1 = nn.Conv3d(outplanes1, int(outplanes1 / 2), kernel_size=kernel, padding=1)
        self.convatt2 = nn.Conv3d(int(outplanes1 / 2), self.num_branches, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # compute attention weights for the second layer
        feature = x.detach()
        att = self.relu(self.convatt1(feature))
        att = self.convatt2(att)
        # att = torch.sigmoid(att)
        att = F.softmax(att, dim=1)
        # att = att[:, :, 1:-1, 1:-1, 1:-1]

        # linear combination of different dilation rates
        x1 = self.conv2(x)
        shape = x1.size()
        x1 = self.bn_list2[0](x1) * att[:, 0:1, :, :, :].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list2[i](x2)
            x1 += x2 * att[:, i:(i + 1), :, :, :].expand(shape)

        #if self.downsample is not None:
            # residual = self.downsample(residual)

        x = x1 + residual
        x = self.relu(x)
        return x


class ResBlockSegse(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, net_mode='3d'):
        super(ResBlockSegse, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.conv1 = conv(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.expan_channels = self.out_channels*2
        self.expansion_conv = conv(self.out_channels, self.expan_channels, 1)
        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)

        self.conv2 = conv(self.expan_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = bn(out_channels)
        self.drop_block = DropBlock3D(block_size=4, drop_prob=0.1)

        if in_channels != out_channels:
            self.res_conv = conv(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res = self.res_conv(x)
        else:
            res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.expansion_conv(x)
        x = self.relu(x)
        seg_x = self.segse_block(x)
        x = x*seg_x

        x = self.conv2(x)
        x = self.drop_block(x)
        x = self.bn2(x)

        out = x + res
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, net_mode='3d'):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.conv1 = conv(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = bn(out_channels)
        self.drop_block = DropBlock3D(block_size=4, drop_prob=0.1)

        if in_channels != out_channels:
            self.res_conv = conv(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res = self.res_conv(x)
        else:
            res = x
        x = self.conv1(x)
        x = self.drop_block(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        out = x + res
        out = self.relu(out)

        return out


class CBAM_Module(nn.Module):
    def __init__(self, dim, in_channels, ratio, kernel_size):
        super(CBAM_Module, self).__init__()

        self.avg_pool = getattr(nn, "AdaptiveAvgPool{0}d".format(dim))(1)
        self.max_pool = getattr(nn, "AdaptiveMaxPool{0}d".format(dim))(1)
        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        self.fc1 = conv_fn(in_channels, in_channels // ratio, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = conv_fn(in_channels // ratio, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = conv_fn(2, 1, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # Channel attention module:（Mc(f) = σ(MLP(AvgPool(f)) + MLP(MaxPool(f)))）
        module_input = x
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = self.sigmoid(avg + mx)
        x = module_input * x

        # Spatial attention module:Ms (f) = σ( f7×7( AvgPool(f) ; MaxPool(F)] )))
        module_input = x
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((avg, mx), dim=1)
        x = self.sigmoid(self.conv(x))
        x = module_input * x
        return x


class ResBlockAtte(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, net_mode='3d'):
        super(ResBlockAtte, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.conv1 = conv(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = bn(out_channels)
        # self.drop_block = DropBlock3D(block_size=7, drop_prob=0.1)

        if in_channels != out_channels:
            self.res_conv = conv(in_channels, out_channels, 1, stride=stride)

        # Channel and Spatial Attention(CBAM)
        self.cbam = CBAM_Module(dim=3, in_channels=out_channels, ratio=2, kernel_size=7)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res = self.res_conv(x)
        else:
            res = x
        x = self.conv1(x)
        # x = self.drop_block(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Channel and Spatial Attention(CBAM)
        x = self.cbam(x)

        out = x + res
        out = self.relu(out)

        return out


class channelAtte(nn.Module):
    def __init__(self, dim, in_channels, ratio):
        super(channelAtte, self).__init__()

        self.avg_pool = getattr(nn, "AdaptiveAvgPool{0}d".format(dim))(1)
        self.max_pool = getattr(nn, "AdaptiveMaxPool{0}d".format(dim))(1)
        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        self.fc1 = conv_fn(in_channels, in_channels // ratio, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = conv_fn(in_channels // ratio, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module:（Mc(f) = σ(MLP(AvgPool(f)) + MLP(MaxPool(f)))）
        module_input = x
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = self.sigmoid(avg + mx)
        x = module_input * x

        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2, net_mode='2d'):
        super(SegSEBlock, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
        else:
            conv = None

        self.in_channels = in_channels
        self.rate = rate
        #
        self.dila_conv = conv(self.in_channels, self.in_channels // self.rate, 3, padding=2, dilation=self.rate)
        self.conv1 = conv(self.in_channels // self.rate, self.in_channels, 1)

    def forward(self, input):
        x = self.dila_conv(input)
        x = self.conv1(x)
        x = nn.Sigmoid()(x)

        return x


class RecombinationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='3d'):
        super(RecombinationBlock, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = conv(self.in_channels, self.expan_channels, 1)
        self.skip_conv = conv(self.in_channels, self.out_channels, 1)
        self.zoom_conv = conv(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = conv(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)

        self.drop_block = DropBlock3D(block_size=4, drop_prob=0.1)


    def forward(self, input):
        x = self.expansion_conv(input)

        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6()(x)
            x = self.norm_conv(x)
            x = self.drop_block(x)

        se_x = self.segse_block(x)

        x = x * se_x

        x = self.zoom_conv(x)

        skip_x = self.skip_conv(input)
        #x = F.dropout(x, 0.2)
        out = x + skip_x

        return out


class RecombinationBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3, net_mode='3d'):
        super(RecombinationBlock1, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = None
            bn = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = conv(self.in_channels, self.expan_channels, 1)
        self.skip_conv = conv(self.in_channels, self.out_channels, 1)
        self.zoom_conv = conv(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = conv(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels, net_mode=net_mode)

        #self.drop_block = DropBlock3D(block_size=7, drop_prob=0.1)


    def forward(self, input):
        x = self.expansion_conv(input)

        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6()(x)
            x = self.norm_conv(x)


        se_x = self.segse_block(x)

        x = x * se_x

        x = self.zoom_conv(x)

        skip_x = self.skip_conv(input)
        #x = self.drop_block(x)
        #x = F.dropout(x, 0.2)
        out = x + skip_x

        return out


class Up(nn.Module):
    def __init__(self, down_in_channels, in_channels, out_channels, conv_block, interpolation=True, net_mode='3d'):
        super(Up, self).__init__()

        if net_mode == '2d':
            inter_mode = 'bilinear'
            trans_conv = nn.ConvTranspose2d
        elif net_mode == '3d':
            inter_mode = 'trilinear'
            trans_conv = nn.ConvTranspose3d
        else:
            inter_mode = None
            trans_conv = None

        if interpolation == True:
            self.up = nn.Upsample(scale_factor=2, mode=inter_mode, align_corners=True)
            # F.interpolate 上采样
        else:
            self.up = trans_conv(down_in_channels, down_in_channels, 2, stride=2)

        # self.conv = RecombinationBlock(down_in_channels + in_channels, out_channels, net_mode=net_mode)
        self.conv = ResBlock(down_in_channels + in_channels, out_channels, net_mode=net_mode)

        # self.atten = Attention_block(down_in_channels, in_channels, out_channels)

    def forward(self, down_x, x):
        up_x = self.up(down_x)
        # up_x = self.atten(up_x, x)
        x = torch.cat((up_x, x), dim=1)

        x = self.conv(x)

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block, net_mode='2d'):
        super(Down, self).__init__()
        if net_mode == '2d':
            maxpool = nn.MaxPool2d
        elif net_mode == '3d':
            maxpool = nn.MaxPool3d
        else:
            maxpool = None

        if conv_block == RecombinationBlock:
            # self.conv = RecombinationBlock(in_channels, out_channels, net_mode=net_mode)
            self.conv = ResBlock(in_channels, out_channels, net_mode=net_mode)
        else:
            self.conv = conv_block(in_channels, out_channels, net_mode=net_mode)
        self.down = maxpool(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        out = self.down(x)

        return x, out


class UNet(nn.Module):
    def __init__(self, in_channels, filter_num_list, class_num, conv_block, net_mode='3d', kernel_size=7):
        global maxpool, inter_mode
        super(UNet, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            maxpool = nn.MaxPool3d
            inter_mode = 'trilinear'
        else:
            conv = None

        self.inc = ResBlock(in_channels, 16)
        self.downconv1 = ResBlock(16, 32)
        self.down = maxpool(2, stride=2)

        self.downconv2 = ResBlock(32, 48)
        self.downconv3 = ResBlock(48, 64)
        self.downconv4 = ResBlock(64, 96)
        # self.downconv5 = RecombinationBlock(96, 128)

        self.downconv1_1 = ResBlock(32, 32)
        self.downconv2_1 = ResBlock(48, 48)
        self.downconv3_1 = ResBlock(64, 64)
        self.downconv4_1 = ResBlock(96, 96)

        # self.autoDown1 = Autofocus2(16, 32, 32, [0, 4, 8, 12], [2, 4, 8, 12], 4)
        # self.autoDown2 = Autofocus2(32, 64, 48, [0, 4, 8, 12], [2, 4, 8, 12], 3)
        # self.autoDown3 = Autofocus2(48, 96, 64, [0, 4, 8, 12], [2, 4, 8, 12], 2)
        # self.autoDown4 = Autofocus2(64, 128, 96, [0, 4, 8, 12], [2, 4, 8, 12],2)

        self.up = nn.Upsample(scale_factor=2, mode=inter_mode, align_corners=True)
        # self.upconv4 = RecombinationBlock(128+96, 128)
        # self.upconv3 = RecombinationBlock(128+64, 64)
        self.upconv4 = ResBlock(128 + 96, 96)
        self.upconv3 = ResBlock(96 + 64, 64)
        self.upconv2 = ResBlock(64 + 48, 48)
        self.upconv1 = ResBlock(48 + 32, 32)

        self.upconv4_1 = ResBlock(96, 96)
        self.upconv3_1 = ResBlock(64, 64)
        self.upconv2_1 = ResBlock(48, 48)
        self.upconv1_1 = ResBlock(32, 32)

        # self.auto33 = Autofocus2(32,64,32,[0,4,8,12],[2,4,8,12],4)
        # self.auto0 = Autofocus_single(32, 64, 32, [0, 4, 6, 8], [2, 4, 6, 8], 3)
        # self.auto1 = Autofocus_single(48, 96, 48, [0, 4, 6, 8], [2, 4, 6, 8], 2)
        # self.auto2 = Autofocus_single(64, 128, 64, [0, 4, 6, 8], [2, 4, 6, 8], 1)

        # down       [32, 48, 64, 96, 128]
        # self.down1 = Down(16, filter_num_list[0], conv_block=conv_block, net_mode=net_mode)
        # self.down2 = Down(filter_num_list[0], filter_num_list[1], conv_block=ResBlockAtte, net_mode=net_mode)
        # self.down3 = Down(filter_num_list[1], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)
        # self.down4 = Down(filter_num_list[2], filter_num_list[3], conv_block=conv_block, net_mode=net_mode)
        #
        # self.bridge = RecombinationBlock(filter_num_list[3], filter_num_list[4], net_mode=net_mode)

        # # 空洞卷积1
        # self.atr1 = nn.ModuleList([
        #     self.atr_block(32, 32),
        #     self.atr_block(32, 32),
        #     self.atr_block(32, 32),
        #     self.atr_block(32, 32)
        # ])
        #
        # # 空洞卷积2
        # self.atr2 = nn.ModuleList([
        #     self.atr_block(48, 48),
        #     self.atr_block(48, 48),
        #     self.atr_block(48, 48)
        # ])
        #
        # # 空洞卷积3
        # self.atr3 = nn.ModuleList([
        #     self.atr_block(64, 64),
        #     self.atr_block(64, 64)
        # ])

        # 空洞卷积4
        # self.atr4 = nn.ModuleList([
        #     self.atr_block(96, 96)
        # ])

        # 残差注意力机制
        # self.resat1 = Resat1()
        # self.resat2 = Resat2()
        # self.resat3 = Resat3()
        # self.resat4 = Resat4()

        # up
        # self.up1 = Up(filter_num_list[4], filter_num_list[3], filter_num_list[3], conv_block=conv_block, net_mode=net_mode)
        # self.up2 = Up(filter_num_list[3], filter_num_list[2], filter_num_list[2], conv_block=conv_block, net_mode=net_mode)
        # self.up3 = Up(filter_num_list[2], filter_num_list[1], filter_num_list[1], conv_block=conv_block, net_mode=net_mode)
        # self.up4 = Up(filter_num_list[1], filter_num_list[0], filter_num_list[0], conv_block=conv_block, net_mode=net_mode)
        self.conv = ResBlock(96, 128, 1)
        self.class_conv = conv(32, class_num, 1)

        self.atten1 = Attention_block(48, 32, 32)
        self.atten2 = Attention_block(64, 48, 48)
        self.atten3 = Attention_block(96, 64, 64)
        self.atten4 = Attention_block(128, 96, 96)

        # self.convend = ResBlock(filter_num_list[0] + filter_num_list[0], filter_num_list[0], net_mode=net_mode)

        # def atr_block(self,in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False):
        #     layer = nn.Sequential(
        #         nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
        #         nn.LeakyReLU(0.2)
        #     )
        #     return layer
        self.Unet2 = UNet2(3, 2,net_mode='3d')

    def forward(self, input, GM):

        x = input

        xi = self.inc(x)

        x1 = self.downconv1(xi)
        x1 = self.downconv1_1(x1)

        x = self.down(x1)
        #xc1 = self.auto0(x1)

        x2 = self.downconv2(x)
        x2 = self.downconv2_1(x2)

        x = self.down(x2)
        #xc2 = self.auto1(x2)

        x3 = self.downconv3(x)
        x3 = self.downconv3_1(x3)

        x = self.down(x3)
        #xc3 = self.auto2(x3)

        x4 = self.downconv4(x)
        x4 = self.downconv4_1(x4)

        x = self.down(x4)

        # x = self.downconv5(x)
        x5 = self.conv(x)

        upx = self.up(x5)
        xa4 = self.atten4(upx, x4)
        x = torch.cat((upx, xa4), dim=1)
        x = self.upconv4(x)
        x = self.upconv4_1(x)

        upx = self.up(x)
        xa3 = self.atten3(upx, x3)
        x = torch.cat((upx, xa3), dim=1)
        x = self.upconv3(x)
        x = self.upconv3_1(x)

        upx = self.up(x)
        xa2 = self.atten2(upx, x2)
        x = torch.cat((upx, xa2), dim=1)
        x = self.upconv2(x)
        x = self.upconv2_1(x)

        upx = self.up(x)
        xa1 = self.atten1(upx, x1)
        x = torch.cat((upx, xa1), dim=1)
        x = self.upconv1(x)
        x = self.upconv1_1(x)

        #
        # x1 = self.auto0(x)
        #
        # conv1, x = self.down1(x)
        #
        # # 第一层的空洞卷积
        # # a1 = self.atr1[0](conv1)
        # # a1 = self.atr1[1](a1)
        # # a1 = self.atr1[2](a1)
        # # a1 = self.atr1[3](a1)
        #
        # # # 第一层注意力机制
        # # out1 = self.resat4(a1)
        #
        # conv1 = self.auto1(conv1)
        #
        # conv2, x = self.down2(x)
        #
        # # 第二层的空洞卷积
        # # a2 = self.atr2[0](conv2)
        # # a2 = self.atr2[1](a2)
        # # a2 = self.atr2[2](a2)
        #
        # # # 第二层注意力机制
        # # out2 = self.resat3(a2)
        #
        # # conv2 = self.auto2(conv2)
        #
        # conv3, x = self.down3(x)
        #
        # # 第三层的空洞卷积
        # # a3 = self.atr3[0](conv3)
        # # a3 = self.atr3[1](a3)
        #
        # # # 第三层注意力机制
        # # out3 = self.resat2(a3)
        #
        #
        # #
        # conv4, x = self.down4(x)
        #
        # # 第四层的空洞卷积
        # # a4 = self.atr4[0](conv4)
        #
        # # # 第四层注意力机制
        # # out4 = self.resat1(a4)
        #
        # x = self.bridge(x)
        #
        #
        # x = self.up1(x, conv4)
        # #
        # x = self.up2(x, conv3)
        #
        # x = self.up3(x, conv2)
        #
        # up_x = self.up(x)
        # up_x = self.atten(up_x, conv1)
        # x = torch.cat((up_x, conv1), dim=1)
        # x = self.convend(x)
        # # x = self.up4(x, conv1)
        #
        # x = self.conv(x)
        # x = x + x1

        x = self.class_conv(x)

        prob = nn.Softmax(1)(x)

        prob1 = x[:, 1:, :, :, :]
        prob1 = (prob1-torch.min(prob1))/(torch.max(prob1)-torch.min(prob1))

        x = self.Unet2(prob1, GM, x1, x2, x3, x4, x5)

        return prob, x


def main():
    #torch.cuda.set_device(1)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    model = UNet(1, [32, 48, 64, 96, 128], 2, conv_block=RecombinationBlock, net_mode='3d').to(device)
    #model = UNet(1, [32, 64, 128, 256, 512], 3, net_mode='3d').to(device)
    x = torch.rand(1, 1, 24, 24, 48)
    prob = torch.rand(1, 1, 24, 24, 48)
    # x = F.pad(x, pad=(1, 1, 1, 1, 0, 0), mode="constant", value=0)
    x = x.to(device)
    prob = prob.to(device)
    model.forward(x)


if __name__ == '__main__':
    main()
