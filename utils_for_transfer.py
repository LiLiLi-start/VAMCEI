import math
from tempfile import tempdir
import cv2
from turtle import forward
import torch
from torch import nn
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import time
import random
from skimage import transform
import torch.nn.init as init
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image


def init_conv(conv):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1),
                                      )

    def forward(self, x):
        actition = self.activate(x)
        out = torch.mul(x, actition)

        return out


class Self_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_channel

        self.f = nn.Conv2d(in_channels=in_channel,
                           out_channels=in_channel // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channel,
                           out_channels=in_channel // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channel,
                           out_channels=in_channel//8, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channel//8,
                           out_channels=in_channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width *
                           height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width *
                           height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width *
                           height)  # B * (C//8) * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * (C//8) * (W * H)
        self_attetion = self_attetion.view(
            m_batchsize, -1, width, height)  # B * (C//8) * W * H

        self_attetion = self.v(self_attetion)   # B * C * W * H

        out = self.gamma * self_attetion  # +x

        return out


class VAE(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAE, self).__init__()
        self.feat = 16
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convt1 = nn.ConvTranspose2d(
            self.feat*64, self.feat*32, kernel_size=2, stride=2)
        self.convt2 = nn.ConvTranspose2d(
            self.feat*32, self.feat*16, kernel_size=2, stride=2)
        self.convt3 = nn.ConvTranspose2d(
            self.feat*16, self.feat*8, kernel_size=2, stride=2)
        self.convt4 = nn.ConvTranspose2d(
            self.feat*8, self.feat*4, kernel_size=2, stride=2)

        self.conv_seq1 = nn.Sequential(nn.Conv2d(1, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*4),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(
            self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.InstanceNorm2d(self.feat*4),
            nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(
                                           self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*16),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*16, self.feat*16,
                                                 kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*16),
                                       nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*32, self.feat*32,
                                                 kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True))
        self.conv_seq5 = nn.Sequential(nn.Conv2d(self.feat*32, self.feat*64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*64, self.feat*64,
                                                 kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*64),
                                       nn.ReLU(inplace=True))

        self.deconv_seq1 = nn.Sequential(nn.Conv2d(self.feat*64, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*32),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(
                                             self.feat*32, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*32),
                                         nn.ReLU(inplace=True))
        self.deconv_seq2 = nn.Sequential(nn.Conv2d(self.feat*32, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*16),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(
                                             self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*16),
                                         nn.ReLU(inplace=True),
                                         )

        self.down4fc1 = nn.Sequential(Spatial_Attention(self.feat*16),
                                      nn.InstanceNorm2d(self.feat*16),
                                      nn.Tanh())
        self.down4fc2 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(self.feat*16),
                                      nn.Tanh())

        self.segdown4_seq = nn.Sequential(
            # nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*16),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5),
            # nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*16),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5),
            # nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*16),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5),
            # nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*16),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5),
            nn.Conv2d(self.feat*16, 4, kernel_size=KERNEL, padding=PADDING),

        )

        self.deconv_seq3 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*8),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(
                                             self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*8),
                                         nn.ReLU(inplace=True))

        self.down2fc1 = nn.Sequential(Spatial_Attention(self.feat*8),
                                      nn.InstanceNorm2d(self.feat*8),
                                      nn.Tanh())
        self.down2fc2 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(self.feat*8),
                                      nn.Tanh())
        self.segdown2_seq = nn.Sequential(
            # nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*8),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5),
            # nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*8),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*8),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5),
            # nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*8),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5),

            nn.Conv2d(self.feat*8, 4, kernel_size=KERNEL, padding=PADDING),
        )

        self.deconv_seq4 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*4),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(
                                             self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*4),
                                         nn.ReLU(inplace=True),)

        self.fc1 = nn.Sequential(Spatial_Attention(self.feat*4),
                                 nn.InstanceNorm2d(self.feat*4),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(self.feat*4),
                                 nn.Tanh())

        self.deconv_seq5 = nn.Sequential(
            # nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*4),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*4),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*4),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            # nn.InstanceNorm2d(self.feat*4),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*4,
                      kernel_size=KERNEL, padding=PADDING),
            nn.InstanceNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, 4, kernel_size=KERNEL, padding=PADDING)
        )
        self.soft = nn.Softmax2d()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.segfusion = nn.Sequential(nn.Conv2d(4*3, 12, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(12),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(4 * 3, 4, kernel_size=KERNEL, padding=PADDING),)

    def reparameterize(self, mu, logvar, gate):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp*gate
        return z

    def bottleneck(self, h, gate):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneckdown2(self, h, gate):
        mu, logvar = self.down2fc1(h), self.down2fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneckdown4(self, h, gate):
        mu, logvar = self.down4fc1(h), self.down4fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def encode(self, x, gate):
        out1 = self.conv_seq1(x)
        out2 = self.conv_seq2(self.maxpool(out1))
        out3 = self.conv_seq3(self.maxpool(out2))
        out4 = self.conv_seq4(self.maxpool(out3))
        out5 = self.conv_seq5(self.maxpool(out4))

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5), out4), 1))
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1), out3), 1))
        feat_down4, down4_mu, down4_logvar = self.bottleneckdown4(deout2, gate)
        segout_down4 = self.segdown4_seq(feat_down4)
        pred_down4 = self.soft(segout_down4)
        deout3 = self.deconv_seq3(
            torch.cat((self.convt3(feat_down4), out2), 1))
        feat_down2, down2_mu, down2_logvar = self.bottleneckdown2(deout3, gate)
        segout_down2 = self.segdown2_seq(feat_down2)
        pred_down2 = self.soft(segout_down2)
        deout4 = self.deconv_seq4(
            torch.cat((self.convt4(feat_down2), out1), 1))
        z, mu, logvar = self.bottleneck(deout4, gate)
        return z, mu, logvar, pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar, pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar, out5

    def forward(self, x, gate):
        z, mu, logvar, pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar, pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar, out5 = self.encode(
            x, gate)
        out = self.deconv_seq5(z)
        pred = self.soft(out)
        fusion_seg = self.segfusion(torch.cat(
            (pred, self.upsample2(pred_down2), self.upsample4(pred_down4)), dim=1))

        return fusion_seg, pred, out, z, mu, logvar, pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar, pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar, out5


class VAE_beta(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAE_beta, self).__init__()
        self.feat = 16
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convt1 = nn.ConvTranspose2d(
            self.feat*64, self.feat*32, kernel_size=2, stride=2)
        self.convt2 = nn.ConvTranspose2d(
            self.feat*32, self.feat*16, kernel_size=2, stride=2)
        self.convt3 = nn.ConvTranspose2d(
            self.feat*16, self.feat*8, kernel_size=2, stride=2)
        self.convt4 = nn.ConvTranspose2d(
            self.feat*8, self.feat*4, kernel_size=2, stride=2)

        self.conv_seq1 = nn.Sequential(nn.Conv2d(1, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*4),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(
            self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
            nn.InstanceNorm2d(self.feat*4),
            nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(
                                           self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*8),
                                       nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*16),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*16, self.feat*16,
                                                 kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*16),
                                       nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*32, self.feat*32,
                                                 kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*32),
                                       nn.ReLU(inplace=True))
        self.conv_seq5 = nn.Sequential(nn.Conv2d(self.feat*32, self.feat*64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.feat*64, self.feat*64,
                                                 kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(self.feat*64),
                                       nn.ReLU(inplace=True))

        self.deconv_seq1 = nn.Sequential(nn.Conv2d(self.feat*64, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*32),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(
                                             self.feat*32, self.feat*32, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*32),
                                         nn.ReLU(inplace=True))
        self.deconv_seq2 = nn.Sequential(nn.Conv2d(self.feat*32, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*16),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(
                                             self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*16),
                                         nn.ReLU(inplace=True),
                                         )

        self.down4fc1 = nn.Sequential(Spatial_Attention(self.feat*16),
                                      nn.InstanceNorm2d(self.feat*16),
                                      nn.Tanh())
        self.down4fc2 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*16, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(self.feat*16),
                                      nn.Tanh())

        self.segdown4_seq = nn.Sequential(
            nn.Conv2d(self.feat*16, 4, kernel_size=KERNEL, padding=PADDING),
        )

        self.deconv_seq3 = nn.Sequential(nn.Conv2d(self.feat*16, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*8),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(
                                             self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*8),
                                         nn.ReLU(inplace=True))

        self.down2fc1 = nn.Sequential(Spatial_Attention(self.feat*8),
                                      nn.InstanceNorm2d(self.feat*8),
                                      nn.Tanh())
        self.down2fc2 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*8, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(self.feat*8),
                                      nn.Tanh())
        self.segdown2_seq = nn.Sequential(
            nn.Conv2d(self.feat*8, 4, kernel_size=KERNEL, padding=PADDING),)

        self.deconv_seq4 = nn.Sequential(nn.Conv2d(self.feat*8, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*4),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.5),
                                         nn.Conv2d(
                                             self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*4),
                                         nn.ReLU(inplace=True),)

        self.fc1 = nn.Sequential(Spatial_Attention(self.feat*4),
                                 nn.InstanceNorm2d(self.feat*4),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(self.feat*4),
                                 nn.Tanh())

        self.deconv_seq5 = nn.Sequential(nn.Conv2d(self.feat*4, self.feat*4, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(self.feat*4),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(self.feat*4, 4, kernel_size=KERNEL, padding=PADDING))
        self.soft = nn.Softmax2d()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.segfusion = nn.Sequential(nn.Conv2d(4*3, 12, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(12),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(4 * 3, 4, kernel_size=KERNEL, padding=PADDING),)

    def reparameterize(self, mu, logvar, gate):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp*gate
        return z

    def bottleneck(self, h, gate):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneckdown2(self, h, gate):
        mu, logvar = self.down2fc1(h), self.down2fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneckdown4(self, h, gate):
        mu, logvar = self.down4fc1(h), self.down4fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def encode(self, x, gate):
        out1 = self.conv_seq1(x)
        out2 = self.conv_seq2(self.maxpool(out1))
        out3 = self.conv_seq3(self.maxpool(out2))
        out4 = self.conv_seq4(self.maxpool(out3))
        out5 = self.conv_seq5(self.maxpool(out4))

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5), out4), 1))
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1), out3), 1))

        feat_down4, down4_mu, down4_logvar = self.bottleneckdown4(deout2, gate)
        segout_down4 = self.segdown4_seq(feat_down4)
        pred_down4 = self.soft(segout_down4)
        deout3 = self.deconv_seq3(
            torch.cat((self.convt3(feat_down4), out2), 1))

        feat_down2, down2_mu, down2_logvar = self.bottleneckdown2(deout3, gate)
        segout_down2 = self.segdown2_seq(feat_down2)
        pred_down2 = self.soft(segout_down2)
        deout4 = self.deconv_seq4(
            torch.cat((self.convt4(feat_down2), out1), 1))

        z, mu, logvar = self.bottleneck(deout4, gate)
        return z, mu, logvar, pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar, pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar, out5

    def forward(self, x, gate):
        z, mu, logvar, pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar, pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar, out5 = self.encode(
            x, gate)
        out = self.deconv_seq5(z)
        pred = self.soft(out)
        fusion_seg = self.segfusion(torch.cat(
            (pred, self.upsample2(pred_down2), self.upsample4(pred_down4)), dim=1))

        return fusion_seg, pred, out, z, mu, logvar, pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar, pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar, out5


class InfoNet(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(InfoNet, self).__init__()
        self.feat = 16

        self.info_seq = nn.Sequential(nn.Linear(self.feat*64*10*10, self.feat*16),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.feat*16, 6))

    def forward(self, z):
        z = self.info_seq(z.view(z.size(0), -1))
        return z


class VAEDecode(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode, self).__init__()
        self.feat = 16

        self.decoderB = nn.Sequential(
            nn.Conv2d(self.feat*4+4, self.feat*8,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.feat*8, self.feat*8,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.feat*8, self.feat*4,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.feat*4, self.feat*4,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.feat*4, self.feat*2,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.feat*2, self.feat*2,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.feat*2, 1, kernel_size=KERNEL, padding=PADDING),
            # nn.Sigmoid(),
        )

    def forward(self, z, y):
        z = self.decoderB(torch.cat((z, y), dim=1))
        return z


class VAEDecode_down2(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode_down2, self).__init__()
        self.feat = 16

        self.decoderB = nn.Sequential(
            nn.Conv2d(self.feat*8 + 4, self.feat*8,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*8,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*4,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*4,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*2,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, self.feat*2,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, 1, kernel_size=KERNEL, padding=PADDING),
            # nn.Sigmoid(),
        )

    def forward(self, z, y):
        z = self.decoderB(torch.cat((z, y), dim=1))
        return z


class VAEDecode_down4(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(VAEDecode_down4, self).__init__()
        self.feat = 16

        self.decoderB = nn.Sequential(
            nn.Conv2d(self.feat*16 + 4, self.feat*8,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*8,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*8, self.feat*4,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*4,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*4, self.feat*2,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, self.feat*2,
                      kernel_size=KERNEL, padding=PADDING),
            nn.BatchNorm2d(self.feat*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat*2, 1, kernel_size=KERNEL, padding=PADDING),
            # nn.Sigmoid(),
        )

    def forward(self, z, y):
        z = self.decoderB(torch.cat((z, y), dim=1))
        return z


class Discriminator(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(Discriminator, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, dilation=2),  # 190
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, dilation=2),  # 190
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.linear_seq = nn.Sequential(nn.Linear(32*5*5, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(64, 1),
                                        )

    def forward(self, y):
        out = self.decoder(y)
        out = self.linear_seq(out.view(out.size(0), -1))
        out = out.mean()
        return out


class LGE_TrainSet(Dataset):
    def __init__(self, dir, sample_num, times=4):
        self.imgdir = dir+'/LGE/'

        self.imgsname = glob.glob(self.imgdir + '*LGE.nii*')

        imgs = np.zeros((1, 192, 192))
        self.info = []
        self.times = int((45.0 / sample_num) * times)
        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            imgs = np.concatenate((imgs, npimg), axis=0)
            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:, :, :]

    def __getitem__(self, item):
        imgindex, crop_indice = divmod(item, self.times)

        npimg = self.imgs[imgindex, :, :].copy()
        randx = np.random.randint(-16, 16)
        randy = np.random.randint(-16, 16)

        npimg = npimg[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]

        # self.trans = []
        # self.trans.append(A.CenterCrop(192, 192))
        # self.trans.append(A.ShiftScaleRotate(interpolation=Image.BICUBIC))
        # self.trans.append(A.HorizontalFlip())
        # self.trans.append(A.VerticalFlip())
        # self.trans.append(A.CenterCrop(192-randx, 192-randy))
        # self.trans.append(A.Resize(160, 160, interpolation=Image.BICUBIC))
        # # self.trans.append(A.GridDistortion(p=1))
        # self.trans = A.Compose(self.trans)

        # trans_image = self.trans(image=npimg)

        # npimg = trans_image['image']
        # nplab = trans_image['mask']

        # npimg = np.array([npimg])
        # npimg = npimg.astype(np.float32)

        # npimg_o = transform.resize(npimg, (80, 80),
        #                      order=3, mode='edge', preserve_range=True)
        #npimg_resize = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # npimg_down2 = transform.resize(
        #     npimg, (80, 80), order=3, mode='edge', preserve_range=True)
        # npimg_down4 = transform.resize(
        #     npimg, (40, 40), order=3, mode='edge', preserve_range=True)

        self.trans_2 = A.Compose(
            [A.Resize(80, 80, interpolation=cv2.INTER_NEAREST)])
        self.trans_4 = A.Compose(
            [A.Resize(40, 40, interpolation=cv2.INTER_NEAREST)])

        resize_2 = self.trans_2(image=npimg)
        resize_4 = self.trans_4(image=npimg)

        npimg_down2 = resize_2['image']
        # nplab_down2 = resize_2['mask']

        npimg_down4 = resize_4['image']
        # nplab_down4 = resize_4['mask']

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor), torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)
        # return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor)

    def __len__(self):

        return self.imgs.shape[0]*self.times


class LGE_ValSet(Dataset):
    def __init__(self, dir, sample_num):
        self.imgdir = dir+'/LGE_val/'

        self.imgsname = glob.glob(self.imgdir + '*LGE.nii*')

        # imgs = np.zeros((1, 192, 192))
        # labs = np.zeros((1, 192, 192))
        imgs = []
        labs = []
        self.info = []
        self.times = 1

        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            # print(npimg.shape)
            # plt.imshow(npimg[0])
            # plt.show()
            # imgs = np.concatenate((imgs, npimg), axis=0)
            imgs.append(npimg)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * \
                2 + (nplab == 600) * 3

            # labs = np.concatenate((labs, nplab), axis=0)
            labs.append(nplab)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs
        self.labs = labs
        # self.imgs.astype(np.float32)
        # self.labs.astype(np.float32)

    def __getitem__(self, item):
        imgindex, crop_indice = divmod(item, self.times)

        # (bs, slices, 192, 192)
        npimg = self.imgs[imgindex]
        nplab = self.labs[imgindex]

        npimg.astype(np.float32)
        nplab.astype(np.float32)

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        # randx = np.random.randint(-16, 16)
        # randy = np.random.randint(-16, 16)
        randx = 0
        randy = 0
        npimg = npimg[:, 96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        nplab = nplab[:, 96+randx-80:96+randx+80, 96+randy-80:96+randy+80]

        # print(npimg.shape)
        # print(nplab.shape)

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        npimg_down2 = transform.resize(
            npimg, (80, 80), order=3, mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(
            npimg, (40, 40), order=3, mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(
            nplab, (80, 80), order=0, mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(
            nplab, (40, 40), order=0, mode='edge', preserve_range=True)

        # return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor), torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)
        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor)

    def __len__(self):

        return len(self.imgs)*self.times


class LGE_TestSet(Dataset):
    def __init__(self, dir, sample_num):
        self.imgdir = dir+'/LGE_test/'

        self.imgsname = glob.glob(self.imgdir + '*LGE.nii*')

        # imgs = np.zeros((1, 192, 192))
        # labs = np.zeros((1, 192, 192))
        imgs = []
        labs = []
        self.info = []
        self.times = 1

        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            # print(npimg.shape)
            # plt.imshow(npimg[0])
            # plt.show()
            # imgs = np.concatenate((imgs, npimg), axis=0)
            imgs.append(npimg)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * \
                2 + (nplab == 600) * 3

            # labs = np.concatenate((labs, nplab), axis=0)
            labs.append(nplab)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs
        self.labs = labs
        # self.imgs.astype(np.float32)
        # self.labs.astype(np.float32)

    def __getitem__(self, item):
        imgindex, crop_indice = divmod(item, self.times)

        # (bs, slices, 192, 192)
        npimg = self.imgs[imgindex]
        nplab = self.labs[imgindex]

        npimg.astype(np.float32)
        nplab.astype(np.float32)

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        # randx = np.random.randint(-16, 16)
        # randy = np.random.randint(-16, 16)
        randx = 0
        randy = 0
        npimg = npimg[:, 96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        nplab = nplab[:, 96+randx-80:96+randx+80, 96+randy-80:96+randy+80]

        # print(npimg.shape)
        # print(nplab.shape)

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        npimg_down2 = transform.resize(
            npimg, (80, 80), order=3, mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(
            npimg, (40, 40), order=3, mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(
            nplab, (80, 80), order=0, mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(
            nplab, (40, 40), order=0, mode='edge', preserve_range=True)

        # return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor), torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)
        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor)

    def __len__(self):

        return len(self.imgs)*self.times


class C0_TrainSet(Dataset):
    def __init__(self, dir, sample_num, val=False, times=4):
        if val:
            self.imgdir = dir+'/C0_val/'
        else:
            self.imgdir = dir+'/C0/'

        self.imgsname = glob.glob(self.imgdir + '*C0.nii*')

        imgs = np.zeros((1, 192, 192))
        labs = np.zeros((1, 192, 192))
        self.info = []
        # self.times = 1
        self.times = int((35.0 / sample_num) * times)
        if val:
            self.times = 1
        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            # print(npimg.shape)
            # plt.imshow(npimg[0])
            # plt.show()
            imgs = np.concatenate((imgs, npimg), axis=0)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * \
                2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:, :, :]
        self.labs = labs[1:, :, :]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)

    def __getitem__(self, item):
        imgindex, crop_indice = divmod(item, self.times)

        npimg = self.imgs[imgindex, :, :].copy()
        nplab = self.labs[imgindex, :, :].copy()
        # print(npimg.shape)

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        randx = np.random.randint(-16, 16)
        randy = np.random.randint(-16, 16)
        npimg = npimg[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        nplab = nplab[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]

        # self.trans = []
        # self.trans.append(A.CenterCrop(192, 192))
        # self.trans.append(A.ShiftScaleRotate(interpolation=Image.BICUBIC))
        # self.trans.append(A.HorizontalFlip())
        # self.trans.append(A.VerticalFlip())
        # self.trans.append(A.CenterCrop(192-randx, 192-randy))
        # self.trans.append(A.Resize(160, 160, interpolation=Image.BICUBIC))
        # # self.trans.append(A.GridDistortion(p=1))
        # self.trans = A.Compose(self.trans)

        # trans_image = self.trans(image=npimg, mask=nplab)

        # npimg = trans_image['image']
        # nplab = trans_image['mask']

        # npimg = np.array([npimg])
        # npimg = npimg.astype(np.float32)

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        self.trans_2 = A.Compose(
            [A.Resize(80, 80, interpolation=cv2.INTER_NEAREST)])
        self.trans_4 = A.Compose(
            [A.Resize(40, 40, interpolation=cv2.INTER_NEAREST)])

        resize_2 = self.trans_2(image=npimg, mask=nplab)
        resize_4 = self.trans_4(image=npimg, mask=nplab)

        npimg_down2 = resize_2['image']
        nplab_down2 = resize_2['mask']

        npimg_down4 = resize_4['image']
        nplab_down4 = resize_4['mask']

        # npimg_down2 = transform.resize(
        #     npimg, (80, 80), order=3, mode='edge', preserve_range=True)
        # npimg_down4 = transform.resize(
        #     npimg, (40, 40), order=3, mode='edge', preserve_range=True)

        # nplab_down2 = transform.resize(
        #     nplab, (80, 80), order=0, mode='edge', preserve_range=True)
        # nplab_down4 = transform.resize(
        #     nplab, (40, 40), order=0, mode='edge', preserve_range=True)

        # plt.subplot(1,3,1)
        # plt.imshow(nplab)
        # plt.subplot(1,3,2)
        # plt.imshow(nplab_down2)
        # plt.subplot(1,3,3)
        # plt.imshow(nplab_down4)
        # plt.show()

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor), torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)
        # return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*self.times


class C0_ValSet(Dataset):
    def __init__(self, dir, sample_num):
        self.imgdir = dir+'/C0_val/'

        self.imgsname = glob.glob(self.imgdir + '*C0.nii*')

        # imgs = np.zeros((1, 192, 192))
        # labs = np.zeros((1, 192, 192))
        imgs = []
        labs = []
        self.info = []
        self.times = 1

        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            # print(npimg.shape)
            # plt.imshow(npimg[0])
            # plt.show()
            # imgs = np.concatenate((imgs, npimg), axis=0)
            imgs.append(npimg)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * \
                2 + (nplab == 600) * 3

            # labs = np.concatenate((labs, nplab), axis=0)
            labs.append(nplab)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs
        self.labs = labs
        # self.imgs.astype(np.float32)
        # self.labs.astype(np.float32)

    def __getitem__(self, item):
        imgindex, crop_indice = divmod(item, self.times)

        # (bs, slices, 192, 192)
        npimg = self.imgs[imgindex]
        nplab = self.labs[imgindex]

        npimg.astype(np.float32)
        nplab.astype(np.float32)

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        # randx = np.random.randint(-16, 16)
        # randy = np.random.randint(-16, 16)
        randx = 0
        randy = 0
        npimg = npimg[:, 96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        nplab = nplab[:, 96+randx-80:96+randx+80, 96+randy-80:96+randy+80]

        # print(npimg.shape)
        # print(nplab.shape)

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        npimg_down2 = transform.resize(
            npimg, (80, 80), order=3, mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(
            npimg, (40, 40), order=3, mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(
            nplab, (80, 80), order=0, mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(
            nplab, (40, 40), order=0, mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor), torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)
        # return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor)

    def __len__(self):

        return len(self.imgs)*self.times


def dice_compute(pred, groundtruth):  # batchsize*channel*W*W
    # for j in range(pred.shape[0]):
    #     for i in range(pred.shape[1]):
    #         if np.sum(pred[j,i,:,:])==0 and np.sum(groundtruth[j,i,:,:])==0:
    #             pred[j, i, :, :]=pred[j, i, :, :]+1
    #             groundtruth[j, i, :, :]=groundtruth[j,i,:,:]+1
    #
    # dice = 2*np.sum(pred*groundtruth,axis=(2,3),dtype=np.float16)/(np.sum(pred,axis=(2,3),dtype=np.float16)+np.sum(groundtruth,axis=(2,3),dtype=np.float16))
    dice = []
    for i in range(4):
        dice_i = 2*(np.sum((pred == i)*(groundtruth == i), dtype=np.float32)+0.0001) / \
            (np.sum(pred == i, dtype=np.float32) +
             np.sum(groundtruth == i, dtype=np.float32)+0.0001)
        dice = dice+[dice_i]

    return np.array(dice, dtype=np.float32)


def IOU_compute(pred, groundtruth):
    iou = []
    for i in range(4):
        iou_i = (np.sum((pred == i)*(groundtruth == i), dtype=np.float32)+0.0001)/(np.sum(pred == i, dtype=np.float32) +
                                                                                   np.sum(groundtruth == i, dtype=np.float32)-np.sum((pred == i)*(groundtruth == i), dtype=np.float32)+0.0001)
        iou = iou+[iou_i]

    return np.array(iou, dtype=np.float32)


def Hausdorff_compute(pred, groundtruth, spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1, 4, 5))
    surface_distance_results = np.zeros((1, 4, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(4):
        pred_i = (pred == i).astype(np.float32)
        if np.sum(pred_i) == 0:
            overlap_results[0, i, :] = 0
            surface_distance_results[0, i, :] = 0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue == i, ITKPred == i)
            overlap_results[0, i,
                            0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0, i,
                            1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0, i,
                            2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0, i,
                            3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0, i,
                            4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue == i, ITKPred == i)

            surface_distance_results[0, i,
                                     0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(
                ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(
                statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(
                ITKPred == i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred == i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(
                statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * \
                sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * \
                sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(
                seg2ref_distance_map)
            seg2ref_distances = list(
                seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(
                ref2seg_distance_map)
            ref2seg_distances = list(
                ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0, i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0, i, 2] = np.median(
                all_surface_distances)
            surface_distance_results[0, i, 3] = np.std(all_surface_distances)
            surface_distance_results[0, i, 4] = np.max(all_surface_distances)

    return overlap_results, surface_distance_results


def multi_dice_iou_compute(pred, label):
    truemax, truearg = torch.max(pred, 1, keepdim=False)
    truearg = truearg.detach().cpu().numpy()
    # nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, \
    #                    truearg == 4, truearg == 5, truearg == 6, truearg == 7), 1)
    nplabs = np.stack((truearg == 0, truearg == 1, truearg ==
                      2, truearg == 3, truearg == 4, truearg == 5), 1)
    # truelabel = (truearg == 0) * 550 + (truearg == 1) * 420 + (truearg == 2) * 600 + (truearg == 3) * 500 + \
    #             (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0

    dice = dice_compute(nplabs, label.cpu().numpy())
    Iou = IOU_compute(nplabs, label.cpu().numpy())

    return dice, Iou


class BalancedBCELoss(nn.Module):
    def __init__(self, target):
        super(BalancedBCELoss, self).__init__()
        self.eps = 1e-6
        weight = torch.tensor([torch.reciprocal(torch.sum(target == 0).float()+self.eps), torch.reciprocal(torch.sum(target == 1).float(
        )+self.eps), torch.reciprocal(torch.sum(target == 2).float()+self.eps), torch.reciprocal(torch.sum(target == 3).float()+self.eps)])
        self.criterion = nn.CrossEntropyLoss(weight)

    def forward(self, output, target):
        loss = self.criterion(output, target)

        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # sig = torch.sum(target, dim=1)
        # n = len(sig.nonzero())

        # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + \
            torch.sum(target, dim=1) + self.smooth

        dice_score = (2 * num + self.smooth) / den
        loss_avg = 1 - dice_score

        return loss_avg


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        bs, num_cls, H, W = predict.shape
        # (1,5,3,256,256)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        predict = torch.softmax(predict, dim=1)

        predict = predict[:, 1:]
        target = target[:, 1:]
        predict = predict.contiguous().view(-1, H, W)
        target = target.contiguous().view(-1, H, W)
        predict = predict.float()
        target = target.float()
        dice_loss = dice(predict, target)

        total_loss = dice_loss.mean()

        return total_loss


# edge loss计算


class edge_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor(
            [[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor(
            [[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        # (2, 3, 3)
        G = G.unsqueeze(1)
        # (2, 1, 3, 3)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

        self.loss_fn = nn.MSELoss()

    def forward(self, predict, ground_truth, num_classes):

        loss = torch.zeros((num_classes))
        predict_edge = predict.clone()
        ground_truth_edge = ground_truth.clone()

        for i in range(num_classes):
            x = self.filter(predict_edge[:, i:i+1])
            x = torch.mul(x, x)
            x = torch.sum(x, dim=1, keepdim=True)
            x = torch.sqrt(x)

            # print(x.shape)

            # plt.imshow(x[0,0].cpu().detach().numpy())
            # plt.show()

            y = self.filter(ground_truth_edge[:, i:i+1])
            y = torch.mul(y, y)
            y = torch.sum(y, dim=1, keepdim=True)
            y = torch.sqrt(y)
            # print(y.shape)

            # plt.subplot(1,2,1)
            # plt.imshow(ground_truth_edge[0,i].cpu().detach().numpy())
            # plt.subplot(1,2,2)

            # plt.imshow(y[0,0].cpu().detach().numpy())
            # plt.show()
            loss[i] = self.loss_fn(x, y)

            # loss[i] = torch.mean(torch.mul(x-y, x-y))

        return loss[1].mean()


class Gaussian_Kernel_Function(nn.Module):
    def __init__(self, std):
        super(Gaussian_Kernel_Function, self).__init__()
        self.sigma = std**2

    def forward(self, fa, fb):
        asize = fa.size()
        bsize = fb.size()

        fa1 = fa.view(-1, 1, asize[1])
        fa2 = fa.view(1, -1, asize[1])

        fb1 = fb.view(-1, 1, bsize[1])
        fb2 = fb.view(1, -1, bsize[1])

        aa = fa1-fa2
        vaa = torch.mean(
            torch.exp(torch.div(-torch.pow(torch.norm(aa, 2, dim=2), 2), self.sigma)))

        bb = fb1-fb2
        vbb = torch.mean(
            torch.exp(torch.div(-torch.pow(torch.norm(bb, 2, dim=2), 2), self.sigma)))

        ab = fa1-fb2
        vab = torch.mean(
            torch.exp(torch.div(-torch.pow(torch.norm(ab, 2, dim=2), 2), self.sigma)))

        loss = vaa+vbb-2.0*vab

        return loss


# 计算q_S(z)与q_T(z)之间的距离
class Gaussian_Distance(nn.Module):
    def __init__(self, kern=1):
        super(Gaussian_Distance, self).__init__()
        self.kern = kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)

    def forward(self, mu_a, logvar_a, mu_b, logvar_b):

        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)
        # var_a = torch.exp(logvar_a)
        # var_b = torch.exp(logvar_b)
        var_a = self.avgpool(torch.exp(logvar_a))/(self.kern*self.kern)
        var_b = self.avgpool(torch.exp(logvar_b))/(self.kern*self.kern)
        # var_a = torch.exp(logvar_a)
        # var_b = torch.exp(logvar_b)

        mu_a1 = mu_a.view(mu_a.size(0), 1, -1)
        mu_a2 = mu_a.view(1, mu_a.size(0), -1)

        var_a1 = var_a.view(var_a.size(0), 1, -1)
        var_a2 = var_a.view(1, var_a.size(0), -1)

        mu_b1 = mu_b.view(mu_b.size(0), 1, -1)
        mu_b2 = mu_b.view(1, mu_b.size(0), -1)
        var_b1 = var_b.view(var_b.size(0), 1, -1)
        var_b2 = var_b.view(1, var_b.size(0), -1)

        vaa = torch.sum(torch.div(torch.exp(torch.mul(torch.div(
            torch.pow(mu_a1-mu_a2, 2), var_a1+var_a2), -0.5)), torch.sqrt(var_a1+var_a2)))

        vab = torch.sum(torch.div(torch.exp(torch.mul(torch.div(
            torch.pow(mu_a1-mu_b2, 2), var_a1+var_b2), -0.5)), torch.sqrt(var_a1+var_b2)))
        vbb = torch.sum(torch.div(torch.exp(torch.mul(torch.div(
            torch.pow(mu_b1-mu_b2, 2), var_b1+var_b2), -0.5)), torch.sqrt(var_b1+var_b2)))

        loss = vaa+vbb-torch.mul(vab, 2.0)

        return loss


class Gaussian_Distance_KL(nn.Module):
    def __init__(self, kern=1):
        super(Gaussian_Distance_KL, self).__init__()
        self.kern = kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)

    def forward(self, mu_a_, logvar_a_, mu_b_, logvar_b_):

        mu_a = mu_a_.clone()
        logvar_a = logvar_a_.clone()
        mu_b = mu_b_.clone()
        logvar_b = logvar_b_.clone()

        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)
        var_a = self.avgpool(torch.exp(logvar_a))/(self.kern*self.kern)
        var_b = self.avgpool(torch.exp(logvar_b))/(self.kern*self.kern)
        # torch.Size([4, 64, 40, 40])

        loss = torch.zeros(mu_a.shape).cuda()
        for i in range(mu_a.size(0)):
            # torch.Size([1, 64, 40, 40])
            mu_a_i = mu_a[i:i+1].clone()
            var_a_i = var_a[i:i+1].clone()
            var_a_i = torch.pow(var_a_i, 0.5)
            # return torch.normal(mu, std)
            esp = torch.randn(*mu_a_i.size()).cuda()
            z = mu_a_i + var_a_i * esp*1.0
            z = z.repeat(mu_a.size(0), 1, 1, 1)


            loss += torch.log(torch.mean(torch.div(torch.exp(-torch.div(torch.pow(z-mu_a, 2),
                                                   2*var_a)), torch.sqrt(var_a)), dim=0)+ 1e-6)
            loss -= torch.log(torch.mean(torch.div(torch.exp(-torch.div(torch.pow(z-mu_b, 2),
                                                   2*var_b)), torch.sqrt(var_b)), dim=0)+ 1e-6)
        # print(loss)

        return torch.sum(loss)


class Gaussian_Distance_KL_Jensen(nn.Module):
    def __init__(self, kern=1):
        super(Gaussian_Distance_KL_Jensen, self).__init__()
        self.kern = kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)
        # self.maxpool = nn.MaxPool2d(kernel_size=self.kern, stride= self.kern)
        # self.eps = 1e-6

    def forward(self, mu_a_, logvar_a_, mu_b_, logvar_b_):

        mu_a = mu_a_.clone()
        logvar_a = logvar_a_.clone()
        mu_b = mu_b_.clone()
        logvar_b = logvar_b_.clone()
        # lanel_one_hot = lanel_one_hot_.clone()

        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)
        # lanel_one_hot = self.maxpool(lanel_one_hot)
        var_a = self.avgpool(torch.exp(logvar_a))/(self.kern*self.kern)
        var_b = self.avgpool(torch.exp(logvar_b))/(self.kern*self.kern)
        
        # torch.Size([1, 64, 40, 40])
        # 类别难度不一致
        # 1.像素级权重设置
        # 2.特征通道级别权重设置

        # label = lanel_one_hot.clone()
        # label = torch.argmax(label, dim=1)
        # # (4, 40, 40)

        # weight = torch.tensor([
        #     torch.reciprocal(torch.sum(label == 0).float()+self.eps), 
        #     torch.reciprocal(torch.sum(label == 1).float()+self.eps), 
        #     torch.reciprocal(torch.sum(label == 2).float()+self.eps), 
        #     torch.reciprocal(torch.sum(label == 3).float()+self.eps)
        # ])
        # # print(weight)
        # weight = weight*100

        # label = torch.unsqueeze(label, dim=1)
        # # (4, 1, 40, 40)
        # label = label[0:1]
        # # (1, 1, 40, 40)
        # label = label.repeat((1, mu_a.shape[1], 1, 1))
        # # (1, 64, 40, 40)



        loss = torch.zeros((1, mu_a.shape[1], mu_a.shape[2], mu_a.shape[3])).cuda()
        for i in range(mu_a.size(0)):
            for j in range(mu_a.size(0)):
                mu_s_i = mu_a[i:i+1]
                var_s_i = var_a[i:i+1]

                mu_t = mu_b[j:j+1]
                var_t = var_b[j:j+1]

                mu_s_j = mu_a[j:j+1]
                var_s_j = var_a[j:j+1]

                loss += (torch.log(torch.div(1.0, torch.mul(var_s_j, math.sqrt(2*math.pi)))+ 1e-6)-torch.div(torch.pow(var_s_i, 2)+torch.pow(mu_s_i-mu_s_j, 2), 2*torch.pow(var_s_j, 2)))

                loss -= (torch.log(torch.div(1.0, torch.mul(var_t, math.sqrt(2*math.pi)))+ 1e-6)-torch.div(torch.pow(var_s_i, 2)+torch.pow(mu_s_i-mu_t, 2), 2*torch.pow(var_t, 2)))
        
        loss /= (mu_a.size(0)*mu_a.size(0))
        
        # loss = torch.where((label == 0), loss*weight[0], loss)
        # loss = torch.where((label == 1), loss*weight[1], loss)
        # loss = torch.where((label == 2), loss*weight[2], loss)
        # loss = torch.where((label == 3), loss*weight[3], loss)


        return torch.sum(loss)


# class Gaussian_Distance(nn.Module):
#     def __init__(self):
#         super(Gaussian_Distance, self).__init__()
#         self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
#
#     def forward(self, mu_a,logvar_a,mu_b,logvar_b):
#         mu_a = self.avgpool(mu_a)
#         mu_b = self.avgpool(mu_b)
#         # var_a = torch.exp(logvar_a)
#         # var_b = torch.exp(logvar_b)
#         var_a = self.avgpool(torch.exp(logvar_a))/4
#         var_b = self.avgpool(torch.exp(logvar_b))/4
#
#
#         mu_a1 = mu_a.view(-1,1,mu_a.size(1))
#         mu_a2 = mu_a.view(1,-1,mu_a.size(1))
#         var_a1 = var_a.view(-1,1,var_a.size(1))
#         var_a2 = var_a.view(1,-1,var_a.size(1))
#
#         mu_b1 = mu_b.view(-1,1,mu_b.size(1))
#         mu_b2 = mu_b.view(1,-1,mu_b.size(1))
#         var_b1 = var_b.view(-1,1,var_b.size(1))
#         var_b2 = var_b.view(1,-1,var_b.size(1))
#
#         vaa = torch.mean(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_a2,2),var_a1+var_a2),-0.5)),torch.sqrt(var_a1+var_a2)))
#         vab = torch.mean(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_b2,2),var_a1+var_b2),-0.5)),torch.sqrt(var_a1+var_b2)))
#         vbb = torch.mean(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_b1-mu_b2,2),var_b1+var_b2),-0.5)),torch.sqrt(var_b1+var_b2)))
#
#         # vaa = torch.mean((mu_a1-mu_a2).pow_(2).div_(var_a1+var_a2).mul_(-0.5).exp_().div_((var_a1+var_a2).sqrt_()))
#         # vab = torch.mean((mu_a1-mu_b2).pow_(2).div_(var_a1+var_b2).mul_(-0.5).exp_().div_((var_a1+var_b2).sqrt_()))
#         # vbb = torch.mean((mu_b1-mu_b2).pow_(2).div_(var_b1+var_b2).mul_(-0.5).exp_().div_((var_b1+var_b2).sqrt_()))
#
#         loss = vaa+vbb-torch.mul(vab,2.0)
#
#         return loss


from pathlib import Path
import nibabel as nib
import logging
from os.path import splitext


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', is_mirror=True, shape=(256, 256), is_scale=True, multiple=5, grayscale=True, osize=286, crop_size=256, convert=True):
        # /home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/fake_B
        self.images_dir = Path(images_dir)
        # /home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/c0gt
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.scale = is_scale
        self.mask_suffix = mask_suffix
        self.is_mirror = is_mirror
        self.crop_h, self.crop_w = shape
        self.multiple = multiple
        self.grayscale = grayscale
        self.osize = osize
        self.crop_size = crop_size
        self.convert = convert
        # create the path list
        self.image_list = []
        for image_file_name in os.listdir(self.images_dir):
            # if int(image_file_name.split('_')[0][-1])%5 != num:
            image_file_path = os.path.join(
                self.images_dir, image_file_name)
            label_file_path = os.path.join(
                self.masks_dir, image_file_name.split('.')[0]+'_manual.nii.gz')
            image_file = np.load(image_file_path, allow_pickle=True)
            label_file = nib.load(label_file_path).get_fdata()
            deep = image_file.shape[-1]
            for i in range(deep):
                self.image_list.append({
                    'image': image_file[:, :, i],
                    'label': label_file[:, :, i],
                })

        # self.transform_list_label = transforms.Compose([
        #     transforms.Resize((256, 256))
        # ])
        if not self.image_list:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.image_list)} examples')

    def __len__(self):
        return len(self.image_list)*self.multiple

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['nii.gz', '.gz']:
            return nib.load(filename).get_data()
        elif ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def truncate(self, MRI):
        # truncate
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))

        idexs = np.argwhere(Hist >= 20)
        idex_min = np.float32(0)
        idex_max = np.float32(idexs[-1, 0])

        # MRI[np.where(MRI <= idex_min)] = idex_min
        MRI[np.where(MRI >= idex_max)] = idex_max
        # MRI = MRI - (idex_max+idex_min)/2
        # MRI = MRI / ((idex_max-idex_min)/2)

        # norm
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI /
                       np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def id2trainId(self, label):
        # left ventricular (LV) blood pool (labelled 500),
        # right ventricular blood pool (600),
        # LV normal myocardium (200),
        shape = label.shape
        label = label[0]
        # print(shape)
        # (1,h,w)
        results_map = np.zeros((4, shape[1], shape[2]))

        LV = (label == 50)
        RV = (label == 100)
        MY = (label == 150)

        background = np.logical_not(LV + RV + MY)

        results_map[0, :, :] = np.where(background, 1, 0)
        results_map[1, :, :] = np.where(LV, 1, 0)
        results_map[2, :, :] = np.where(RV, 1, 0)
        results_map[3, :, :] = np.where(MY, 1, 0)
        # results_map[4, :, :] = np.where(edema, 1, 0)
        # results_map[5, :, :] = np.where(scars, 1, 0)
        return results_map

    def __getitem__(self, index):

        index %= len(self.image_list)

        fake_lge_image = self.image_list[index]['image'].copy()
        label = self.image_list[index]['label'].copy()

        label = np.where((label == 500), 50, label)
        label = np.where((label == 600), 100, label)
        label = np.where((label == 200), 150, label)

        label = Image.fromarray(label)
        label = transforms.Resize(
            (286, 286), interpolation=transforms.InterpolationMode.NEAREST)(label)
        label = transforms.CenterCrop(
            (256, 256))(label)
        label = transforms.ToTensor()(label)
        label = label[0]
        label = label.detach().numpy()

        # plt.subplot(1, 2, 1)
        # plt.imshow(fake_lge_image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(label)
        # plt.show()

        self.transform_list = []

        # self.transform_list.append(
        #     A.Resize(256, 256, interpolation=cv2.INTER_NEAREST))

        # self.transform_list.append(A.ShiftScaleRotate(
        #     interpolation=cv2.INTER_NEAREST, scale_limit=0.3))

        self.transform_list.append(A.HorizontalFlip())
        self.transform_list.append(A.VerticalFlip())

        self.transform_list.append(
            A.Resize(256, 256, interpolation=cv2.INTER_NEAREST))

        # self.transform_list.append(A.ElasticTransform(
        #     p=1, alpha=120, sigma=120 * 0.08, alpha_affine=120 * 0.03))
        # self.transform_list.append(A.GridDistortion(p=1))

        self.trans_2 = A.Compose(
            [A.Resize(80, 80, interpolation=cv2.INTER_NEAREST)])
        self.trans_4 = A.Compose(
            [A.Resize(40, 40, interpolation=cv2.INTER_NEAREST)])

        self.transform_list = A.Compose(self.transform_list)

        transform_list = self.transform_list(image=fake_lge_image, mask=label)
        fake_lge_image = transform_list['image']
        label = transform_list['mask']

        # print(label.shape)

        label = np.array([label])
        label = self.id2trainId(label)
        label = np.argmax(label,axis=0)

        # print(label.shape)



        randx = np.random.randint(-16, 16)
        randy = np.random.randint(-16, 16)
        npimg = fake_lge_image[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        nplab = label[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]


        resize_2 = self.trans_2(image=npimg, mask=nplab)
        resize_4 = self.trans_4(image=npimg, mask=nplab)

        npimg_down2 = resize_2['image']
        nplab_down2 = resize_2['mask']

        npimg_down4 = resize_4['image']
        nplab_down4 = resize_4['mask']

        # plt.subplot(2,3,1)
        # plt.imshow(npimg)
        # plt.subplot(2,3,2)
        # plt.imshow(nplab)
        # plt.subplot(2,3,3)
        # plt.imshow(npimg_down2)
        # plt.subplot(2,3,4)
        # plt.imshow(nplab_down2)
        # plt.subplot(2,3,5)
        # plt.imshow(npimg_down4)
        # plt.subplot(2,3,6)
        # plt.imshow(nplab_down4)
        # plt.show()

        # plt.subplot(1, 2, 1)
        # plt.imshow(fake_lge_image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(label)
        # plt.show()


        # fake_lge_image = np.array([fake_lge_image])
        # label = np.array([label])
        # label = self.id2trainId(label)
        # fake_lge_image = fake_lge_image.astype(np.float32)

        # (1, 256, 256)
        # (4, 256, 256)
        # return fake_lge_image, label
        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(nplab).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor), torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor)



class BasicDatasetOriginal(Dataset):
    def __init__(self, images_dir: str, scale: float = 1.0, mask_suffix: str = '', is_mirror=True, shape=(256, 256), is_scale=True,  grayscale=True, osize=286, crop_size=256, convert=True, multiple=5):
        # /home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/SeglgeVal_Image
        self.images_dir = Path(images_dir)
        # /home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/SeglgeVal_Label
        # self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.mask_suffix = mask_suffix
        self.crop_h, self.crop_w = shape
        self.grayscale = grayscale
        self.crop_size = crop_size
        self.convert = convert
        self.multiple = multiple

        # create the path list
        self.image_list = []
        self.transform_list = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(
                    size=[286, 286], interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))]
        )
        # self.transform_list_label = transforms.Compose(
        #     [
        #         transforms.Resize(
        #             size=[286, 286], interpolation=transforms.InterpolationMode.NEAREST, max_size=None, antialias=None),
        #         transforms.CenterCrop(size=(256, 256)),
        #         transforms.ToTensor()]
        # )
        for image_file_name in os.listdir(self.images_dir):
            # if image_file_name.split('_')[0][-1] != str(num):
            image_file_path = os.path.join(
                self.images_dir, image_file_name)
            # label_file_path = os.path.join(
            #     self.masks_dir, image_file_name.split('LGE')[0]+'LGE_manual.nii.gz')
            image_file = nib.load(image_file_path).get_fdata()
            # label_file = nib.load(label_file_path).get_data()
            deep = image_file.shape[-1]
            for i in range(deep):
                self.image_list.append({
                    'image': image_file[:, :, i],
                    # 'label': label_file[:, :, i],
                })
        if not self.image_list:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.image_list)} examples')

    def __len__(self):
        return len(self.image_list)*self.multiple

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['nii.gz', '.gz']:
            return nib.load(filename).get_data()
        elif ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def truncate(self, MRI):
        # truncate
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))

        idexs = np.argwhere(Hist >= 20)
        idex_min = np.float32(0)
        idex_max = np.float32(idexs[-1, 0])

        # MRI[np.where(MRI <= idex_min)] = idex_min
        MRI[np.where(MRI >= idex_max)] = idex_max
        # MRI = MRI - (idex_max+idex_min)/2
        # MRI = MRI / ((idex_max-idex_min)/2)

        # norm
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI /
                       np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def id2trainId(self, label):
        # left ventricular (LV) blood pool (labelled 500),
        # right ventricular blood pool (600),
        # LV normal myocardium (200),
        shape = label.shape
        label = label[0]
        # print(shape)
        # (1,h,w)
        results_map = np.zeros((4, shape[1], shape[2]))

        LV = (label == 50)
        RV = (label == 100)
        MY = (label == 150)

        background = np.logical_not(LV + RV + MY)

        results_map[0, :, :] = np.where(background, 1, 0)
        results_map[1, :, :] = np.where(LV, 1, 0)
        results_map[2, :, :] = np.where(RV, 1, 0)
        results_map[3, :, :] = np.where(MY, 1, 0)
        return results_map

    def __getitem__(self, index):

        index %= len(self.image_list)

        lge_image = self.image_list[index]['image'].copy()
        # label = self.image_list[index]['label']

        # label = np.where((label == 500), 50, label)
        # label = np.where((label == 600), 100, label)
        # label = np.where((label == 200), 150, label)

        # label = torch.tensor(label)
        # label = transforms.CenterCrop((256, 256))(label)
        # label = label.detach().numpy()

        lge_image = lge_image.astype(np.float)
        lge_image = lge_image/(np.max(lge_image)-np.min(lge_image))*255.0

        lge_image = Image.fromarray(lge_image)
        # label = Image.fromarray(label)

        lge_image = self.transform_list(lge_image)
        # label = self.transform_list_label(label)
        # plt.subplot(1, 2, 1)
        # plt.imshow(lge_image[0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(label[0])
        # plt.show()
        # print(lge_image.shape)
        # print(label.shape)
        lge_image = lge_image.detach().numpy()
        lge_image = lge_image[0]
        # label = label.detach().numpy()
        # label = label[0]

        self.transform_list_A = []

        # self.transform_list_A.append(A.ShiftScaleRotate(
        #     interpolation=cv2.INTER_NEAREST, scale_limit=0.3))

        self.transform_list_A.append(A.HorizontalFlip())
        self.transform_list_A.append(A.VerticalFlip())

        self.transform_list_A.append(
            A.Resize(256, 256, interpolation=cv2.INTER_NEAREST))

        # self.transform_list_A.append(A.ElasticTransform(
        #     p=1, alpha=120, sigma=120 * 0.08, alpha_affine=120 * 0.03))
        # self.transform_list_A.append(A.GridDistortion(p=1))

        self.transform_list_A = A.Compose(self.transform_list_A)

        self.trans_2 = A.Compose(
            [A.Resize(80, 80, interpolation=cv2.INTER_NEAREST)])
        self.trans_4 = A.Compose(
            [A.Resize(40, 40, interpolation=cv2.INTER_NEAREST)])

        transform_list_A = self.transform_list_A(image=lge_image)
        lge_image = transform_list_A['image']
        # label = transform_list_A['mask']



        randx = np.random.randint(-16, 16)
        randy = np.random.randint(-16, 16)
        npimg = lge_image[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        # nplab = label[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]

        resize_2 = self.trans_2(image=npimg)
        resize_4 = self.trans_4(image=npimg)

        npimg_down2 = resize_2['image']
        # nplab_down2 = resize_2['mask']

        npimg_down4 = resize_4['image']
        # nplab_down4 = resize_4['mask']


        # plt.subplot(1,3,1)
        # plt.imshow(npimg)
        # plt.subplot(1,3,2)
        # plt.imshow(npimg_down2)
        # plt.subplot(1,3,3)
        # plt.imshow(npimg_down4)
        # plt.show()


        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor), torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor)


import random
def regionMix(source_image, target_image, source_gt, target_predict):
    # 构造中间域图像
    
    # 掩码张量大小为1/4 source_image
    # (batch_size, 1, 160, 160)
    size = source_image.shape[-1]//2
    # 随机生成掩码张量位置
    region_local_x = random.randint(0,size-1)
    region_local_y = random.randint(0,size-1)
    M = torch.zeros(source_image.shape).cuda()
    M[:, :, region_local_x:region_local_x+size, region_local_y:region_local_y+size] = 1
    mix_image = target_image*M + source_image*(1-M)
    mix_gt = target_predict*M + source_gt*(1-M)

    return mix_image, mix_gt, M