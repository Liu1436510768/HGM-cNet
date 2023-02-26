from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock3D
from utils_T import init_util


class Autofocus_single(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus_single, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.con = nn.Conv3d(inplanes1, outplanes2, kernel_size=1)

        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=2,padding=2)
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

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.drop_block(x)
        x = self.bn2(x)

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
        self.drop_block = DropBlock3D(block_size=7, drop_prob=0.1)

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


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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

        self.drop_block = DropBlock3D(block_size=7, drop_prob=0.1)

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
        out = x + skip_x

        return out


class UNet2(nn.Module):
    def __init__(self, in_channels, class_num, net_mode='3d'):
        global maxpool, inter_mode
        super(UNet2, self).__init__()

        if net_mode == '2d':
            conv = nn.Conv2d
        elif net_mode == '3d':
            conv = nn.Conv3d
            maxpool = nn.MaxPool3d
            inter_mode = 'trilinear'
        else:
            conv = None

        self.inc = ResBlock(1, 16)
        #self.inc1 = conv(1, 2, 1)
        # self.end = conv(40, 32, 1)
        # self.inc4 = conv(in_channels, 16, 1)
        # self.inc5 = conv(in_channels, 16, 1)

        # self.CAtten = channelAtte(3, 40, 2)
        #self.res = conv(40, 40, 1)

        self.down = maxpool(2, stride=2)

        self.downconv1 = ResBlock(16, 32)
        self.downconv2 = ResBlock(32, 48)
        self.downconv3 = ResBlock(48, 64)
        self.downconv4 = ResBlock(64, 96)
        self.downconv5 = ResBlock(96, 128, 1)

        #self.cbam = CBAM_Module(3, 48, 2, 7)

        self.downconv1_1 = ResBlock(32, 32)
        self.downconv2_1 = ResBlock(48, 48)
        self.downconv3_1 = ResBlock(64, 64)
        self.downconv4_1 = ResBlock(96, 96)

        self.up = nn.Upsample(scale_factor=2, mode=inter_mode, align_corners=True)

        self.upconv4 = ResBlock(128+96, 96)
        self.upconv3 = ResBlock(64+96, 64)
        self.upconv2 = ResBlock(64+48, 48)
        self.upconv1 = ResBlock(32 + 48, 32)

        self.upconv1_1 = ResBlock(32, 32)
        self.upconv2_1 = ResBlock(48, 48)
        self.upconv3_1 = ResBlock(64, 64)
        self.upconv4_1 = ResBlock(96, 96)

          #self.auto33 = Autofocus2(32,64,32,[0,4,8,12],[2,4,8,12],4)
        #self.auto = Autofocus_single(8, 16, 8, [0, 4, 8, 12], [2, 4, 8, 12], 4)
        #self.auto0 = Autofocus_single(32, 64, 32, [0, 4, 8, 12], [2, 4, 8, 12], 3)
        #self.auto1 = Autofocus_single(48,96,48,[0,4,8,12],[2,4,8,12],2)
        #self.auto2 = Autofocus_single(64, 128, 64, [0, 4, 8, 12], [2, 4, 8, 12], 1)

        self.conv = conv(128, 128, 1)
        self.class_conv = conv(32, class_num, 1)

        self.atten1 = Attention_block(48, 32, 32)
        self.atten2 = Attention_block(64, 48, 48)
        self.atten3 = Attention_block(96, 64, 64)
        self.atten4 = Attention_block(128, 96, 96)

    def forward(self, pred, GM, xc1, xc2, xc3, xc4, xc5):

        #GM = self.inc1(GM)
        #xcat0 = torch.cat((pred, GM), dim=1)
        xcat1 = pred + GM
        #xcat2 = pred - GM
        #xcat3 = pred * GM
        #xcat = torch.cat((xcat0, xcat1), dim=1)
        #xcat = torch.cat((xcat, xcat2), dim=1)
        #xcat = torch.cat((xcat, xcat3), dim=1)

        x = self.inc(xcat1)

        x1 = self.downconv1(x)
        x1 = x1 + xc1
        #catx = torch.cat((x1c, xc1), dim=1)
        x1 = self.downconv1_1(x1)

        x = self.down(x1)
        #xauto0 = self.auto0(x1)

        x2 = self.downconv2(x)
        x2 = x2 + xc2
        #catx = torch.cat((x2c, xc2), dim=1)
        x2 = self.downconv2_1(x2)


        x = self.down(x2)

        x3 = self.downconv3(x)
        x3 = x3 + xc3
        #catx = torch.cat((x3c, xc3), dim=1)
        x3 = self.downconv3_1(x3)


        x = self.down(x3)
        #
        x4 = self.downconv4(x)
        x4 = x4 + xc4
        x4 = self.downconv4_1(x4)

        #
        x = self.down(x4)
        x5 = self.downconv5(x)
        x5c = x5 + xc5
        #x = self.conv(x5c)

        upx = self.up(x5c)
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

        # x = self.end(x)
        x = self.class_conv(x)

        x = nn.Softmax(1)(x)

        return x


def main():
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2(1, 2).to(device)
    init_util.print_network(model)
    #model = UNet(1, [32, 64, 128, 256, 512], 3, net_mode='3d').to(device)
    x = torch.rand(1, 1, 64, 64, 64)
    # x = F.pad(x, pad=(1, 1, 1, 1, 0, 0), mode="constant", value=0)
    x = x.to(device)
    model.forward(x)


if __name__ == '__main__':
    main()
