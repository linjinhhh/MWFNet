from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

BatchNorm3d = nn.InstanceNorm3d
relu_inplace = True
ActivationFunction = nn.ReLU
BN_MOMENTUM = 0.1



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)




def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class feature_Transition_ker1(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(feature_Transition_ker1, self).__init__()


        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv1', nn.Conv3d(num_output_features, num_output_features, kernel_size=1, stride=1))
        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))

class feature_Transition_ker3(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(feature_Transition_ker3, self).__init__()


        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv1', nn.Conv3d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1))
        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))

class feature_Transition_ker7(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(feature_Transition_ker7, self).__init__()


        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=7, stride=1, padding=3, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv1', nn.Conv3d(num_output_features, num_output_features, kernel_size=7,  stride=1, padding=3))
        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, groups=groups, bias=False, dilation=dilation)


def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x7 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul
        return x


class BasicBlock1x1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock1x1, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = ActivationFunction(inplace=relu_inplace)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class BasicBlock1_orgin(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock1_orgin, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = ActivationFunction(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out



class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock1, self).__init__()
        planes = outplanes//2
        if norm_layer is None:
            norm_layer = BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv13 = conv3x3(inplanes, planes//2, stride)
        self.bn13 = norm_layer(planes//2, momentum=BN_MOMENTUM)
        self.relu13 = ActivationFunction(inplace=relu_inplace)

        self.conv15 = conv5x5(inplanes, planes//2, stride)
        self.bn15 = norm_layer(planes//2, momentum=BN_MOMENTUM)
        self.relu15 = ActivationFunction(inplace=relu_inplace)

        self.conv23 = conv3x3(planes, planes)
        self.bn23 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu23 = ActivationFunction(inplace=relu_inplace)

        self.conv25 = conv5x5(planes, planes)
        self.bn25 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu25 = ActivationFunction(inplace=relu_inplace)

        self.conv1x1 = conv1x1(planes*2, planes*2)
        self.conv1x1_res = conv1x1(inplanes, planes//2)
        self.bn = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = ActivationFunction(inplace=relu_inplace)

        self.downsample = downsample
        self.stride = stride
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()


    def forward(self, x):
        identity = x
        res = self.conv1x1_res(x)
        out3 = self.conv13(x)
        out3 = self.bn13(out3)
        # out3 = out3 + res
        out3 = self.relu13(out3)

        out5 = self.conv15(x)
        out5 = self.bn15(out5)
        # out5 = out5 + res
        out5 = self.relu15(out5)

        out = torch.cat([out3, out5], dim=1)
        out = self.ca(out)*out

        out52 = self.conv25(out)
        out52 = self.bn25(out52)
        out52 = self.relu25(out52)

        out32 = self.conv23(out)
        out32 = self.bn23(out32)
        out32 = self.relu23(out32)

        out = self.conv1x1(torch.cat([out32, out52], dim=1))
        out = self.sa(out)*out
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.bn(out) + identity
        out = self.relu(out)
        return out





class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3x3, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = ActivationFunction(inplace=relu_inplace)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out




class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock7x7, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv7x7(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = ActivationFunction(inplace=relu_inplace)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out















class same_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(same_conv, self).__init__()
        self.same = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm3d(ch_out, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )
    def forward(self, x):
        x = self.same(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, scale = 2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.ConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2),
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm3d(ch_out, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
            )

    def forward(self, x):
        x = self.up(x)
        return x






class transition_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(transition_conv, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm3d(ch_out, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )
    def forward(self, x):
        x = self.transition(x)
        return x

class MWFNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(MWFNet, self).__init__()

        features = init_features
        self.feature1 = BasicBlock1(in_channels, features, downsample=nn.Sequential(conv1x1(in_planes=in_channels, out_planes=features), BatchNorm3d(features, momentum=BN_MOMENTUM)))
        self.feature2 = BasicBlock1(features, features, downsample=nn.Sequential(conv1x1(in_planes=features, out_planes=features), BatchNorm3d(features, momentum=BN_MOMENTUM)))
        self.pool1 = down_conv(features, features)
        self.encoder2 = BasicBlock1_orgin(features*2, features * 2)
        self.pool2 = down_conv(features * 2, features * 2)
        self.encoder3 = BasicBlock1_orgin(features * 2*2, features * 4)
        self.pool3 = down_conv(features * 4, features * 4)
        self.encoder4 = BasicBlock1_orgin(features * 4*2, features * 8)
        self.pool4 = down_conv(features * 8, features * 8)
        self.bottleneck = BasicBlock1(features * 12, features * 16, downsample=nn.Sequential(conv1x1(in_planes=features*12, out_planes=features*16), BatchNorm3d(features*16, momentum=BN_MOMENTUM)))
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = BasicBlock1_orgin((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = BasicBlock1_orgin((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = BasicBlock1_orgin((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = BasicBlock1_orgin(features * 3, features)

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.trans_wave1 = BasicBlock1(2, 64, downsample=nn.Sequential(conv1x1(in_planes=2, out_planes=features), BatchNorm3d(features, momentum=BN_MOMENTUM)))
        self.down1 = down_conv(64, 64)
        self.same1 = same_conv(64, 64)
        self.trans_wave2 = BasicBlock1(2, 64, downsample=nn.Sequential(conv1x1(in_planes=2, out_planes=features), BatchNorm3d(features, momentum=BN_MOMENTUM)))
        self.down2 = down_conv(128, 128)
        self.same2 = same_conv(128, 128)
        self.trans_wave3 = BasicBlock1(2, 128, downsample=nn.Sequential(conv1x1(in_planes=2, out_planes=features*2), BatchNorm3d(features*2, momentum=BN_MOMENTUM)))
        self.down3 = down_conv(256, 256)
        self.same3 = same_conv(256, 256)
        self.trans_de1 = BasicBlock1(512, 64, downsample=nn.Sequential(conv1x1(in_planes=512, out_planes=features), BatchNorm3d(features, momentum=BN_MOMENTUM)))
        self.conv_seg1 = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.up_de1 = nn.ConvTranspose3d(
            features, features, kernel_size=8, stride=8
        )
        self.de3_trans1x1 = BasicBlock1x1(128, 64)
        self.trans_de2 = BasicBlock1(256, 64, downsample=nn.Sequential(conv1x1(in_planes=256, out_planes=features), BatchNorm3d(features, momentum=BN_MOMENTUM)))
        self.conv_seg2 = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.up_de2 = nn.ConvTranspose3d(
            features, features, kernel_size=4, stride=4
        )
        self.de2_trans1x1 = BasicBlock1x1(128, 64)
        self.trans_de3 = BasicBlock1(128, 64, downsample=nn.Sequential(conv1x1(in_planes=128, out_planes=features), BatchNorm3d(features, momentum=BN_MOMENTUM)))
        self.conv_seg3 = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.up_de3 = nn.ConvTranspose3d(
            features, features, kernel_size=2, stride=2
        )
        self.de1_trans1x1 = BasicBlock1x1(128, 64)
        self.dropout5 = nn.Dropout3d(p=0.5, inplace=False)
        self.dropout4 = nn.Dropout3d(p=0.4, inplace=False)
        self.dropout3 = nn.Dropout3d(p=0.3, inplace=False)


    def forward(self, x, wave1, wave2, wave3):
        wave1 = self.trans_wave1(wave1)
        wave1_down = self.down1(wave1)
        wave1 = self.same1(wave1)

        wave2 = self.trans_wave2(wave2)
        wave3 = self.trans_wave3(wave3)
        wave2 = torch.cat([wave1_down, wave2], dim=1)
        wave2 = self.same2(wave2)
        wave2_down = self.down2(wave2)

        wave3 = torch.cat([wave2_down, wave3], dim=1)
        wave3 = self.same3(wave3)
        wave3_down = self.down3(wave3)


        f1 = self.feature1(x)
        enc1 = self.feature2(f1)
        # enc1 = self.encoder1(x)
        enc1_down = self.pool1(enc1)
        enc1_down = torch.cat([wave1, enc1_down], dim=1)
        enc2 = self.encoder2(enc1_down)
        enc2_down = self.pool2(enc2)
        enc2_down = torch.cat([wave2, enc2_down], dim=1)
        enc3 = self.encoder3(enc2_down)
        enc3_down = self.pool3(enc3)
        enc3_down = torch.cat([wave3, enc3_down], dim=1)
        enc4 = self.encoder4(enc3_down)
        enc4_down = self.pool4(enc4)
        enc4_down = torch.cat([wave3_down, enc4_down], dim=1)
        bottleneck = self.bottleneck(enc4_down)
        bottleneck = self.dropout5(bottleneck)
        dec4 = self.upconv4(bottleneck)
        enc4 = self.dropout4(enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec4_trans = self.trans_de1(dec4)
        dec4_up = self.up_de1(dec4_trans)
        dec4_seg = self.conv_seg1(dec4_up)

        enc3 = self.dropout4(enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec3_trans = self.trans_de2(dec3)
        dec3_up = self.up_de2(dec3_trans)
        dec3_up = torch.cat([dec3_up, dec4_up], dim=1)
        dec3_up = self.dropout3(dec3_up)
        dec3_up = self.de3_trans1x1(dec3_up)
        dec3_seg = self.conv_seg2(dec3_up)

        enc2 = self.dropout4(enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec2_trans = self.trans_de3(dec2)
        dec2_up = self.up_de3(dec2_trans)
        dec2_up = torch.cat([dec2_up, dec3_up], dim=1)
        dec2_up = self.dropout3(dec2_up)
        dec2_up = self.de2_trans1x1(dec2_up)
        dec2_seg = self.conv_seg3(dec2_up)

        dec1 = torch.cat((dec1, enc1, dec2_up), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        return outputs, dec4_seg, dec3_seg, dec2_seg



    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )





























































class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm3d(ch_out, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )
    def forward(self, x):
        x = self.down(x)
        return x




class transition_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(transition_conv, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm3d(ch_out, momentum=BN_MOMENTUM),
            ActivationFunction(inplace=relu_inplace)
        )
    def forward(self, x):
        x = self.transition(x)
        return x













