# -*- encoding: utf-8 -*-
'''
@datatime: '18-6-6 上午9:56'

@author: wowjoy
'''

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, useBN=False):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias),
            nn.ReLU()
        )


def add_conv_stage2(dim_in, dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, useBN=False):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(dim_out),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias),
            nn.ReLU()
        )


def upsample(ch_coarse, ch_fine):
    return nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)


class Net(nn.Module):
    def __init__(self, in_channel=3,useBN=True, num_classes=2):
        super(Net, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_channel)
        r = 2
        self.conv1 = add_conv_stage(in_channel, 16 * r, useBN=useBN)
        self.conv2 = add_conv_stage(16 * r, 16 * r, useBN=useBN)
        self.conv3 = add_conv_stage(16 * r, 32 * r, useBN=useBN)
        self.conv4 = add_conv_stage(32 * r, 32 * r, useBN=useBN)
        self.conv5 = add_conv_stage(32 * r, 64 * r, useBN=useBN)
        self.conv6 = add_conv_stage(64 * r, 64 * r, useBN=useBN)

        self.conv7 = add_conv_stage(64 * r, 128 * r, padding=2, dilation=2, useBN=useBN)
        self.conv8 = add_conv_stage(128 * r, 128 * r, padding=4, dilation=4, useBN=useBN)

        self.conv9 = add_conv_stage(128 * r, 256 * r, padding=8, dilation=8, useBN=useBN)
        self.conv10 = add_conv_stage(256 * r, 256 * r, kernel_size=1, padding=0, useBN=useBN)
        # self.conv11  = add_conv_stage(256*r, num_classes, kernel_size=1, padding=0, useBN=useBN)
        self.conv11 = nn.Conv2d(256 * r, num_classes, 1, 1, 0, bias=False)

        self.max_pool1 = nn.MaxPool2d(3, 2, 1)
        self.max_pool2 = nn.MaxPool2d(3, 1, 1)
        self.drop = nn.Dropout(0.5, True)

        self.upsample = upsample(2, 2)

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    print('--------')
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                if m.bias:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        bn_x = self.bn0(x)
        conv1_out = self.conv1(bn_x)
        conv2_out = self.conv2(conv1_out)
        pool1 = self.max_pool1(conv2_out)
        conv3_out = self.conv3(pool1)
        conv4_out = self.conv4(conv3_out)
        pool2 = self.max_pool1(conv4_out)
        conv5_out = self.conv5(pool2)
        conv6_out = self.conv6(conv5_out)
        conv7_out = self.conv6(conv6_out)
        pool3 = self.max_pool2(conv7_out)
        conv8_out = self.conv7(pool3)
        pool4 = self.max_pool2(conv8_out)
        conv9_out = self.conv8(pool4)
        pool5 = self.max_pool2(conv9_out)
        conv10_out = self.drop(self.conv9(pool5))
        conv11_out = self.drop(self.conv10(conv10_out))
        conv12_out = self.conv11(conv11_out)
        up1 = self.upsample(conv12_out)
        up2 = self.upsample(up1)
        return up2


def test():
    net = Net(useBN=True, num_classes=2)
    x = Variable(torch.rand(2, 3, 512, 512))
    y = net(x)
    print(y.size())
