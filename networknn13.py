from __future__ import print_function
import torch.nn as nn
import torch

# follow https://arxiv.org/pdf/1803.05400.pdf

ngpu = 1
use_cuda = True if (torch.cuda.is_available() and ngpu > 0) else False



class Generator(nn.Module):
    def unet_conv(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU()
        )
    def unet_conv_transpose(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def conv_resize(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

        )

    def __init__(self, ngpu, use_cuda):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.use_cuda = use_cuda
        self.conv1 = self.unet_conv(1, 64, 3, 2, 1)
        self.conv2 = self.unet_conv(64, 128, 4, 2, 1)
        self.conv3 = self.unet_conv(128, 256, 4, 2, 1)
        self.conv4 = self.unet_conv(256, 512, 4, 2, 1)
        self.conv5 = self.unet_conv(512, 512, 4, 2, 1)
        self.conv6 = self.unet_conv(512, 512, 4, 2, 1)

        self.conv7 = self.unet_conv_transpose(512, 1024, 4, 2, 1)
        self.conv8 = self.unet_conv_transpose(512, 512, 4, 2, 1)
        self.conv9 = self.unet_conv_transpose(256, 256, 4, 2, 1)
        self.conv10 = self.unet_conv_transpose(128, 128, 4, 2, 1)
        self.conv11 = self.unet_conv_transpose(128, 64, 4, 2, 1)
        self.conv12 = self.unet_conv_transpose(128, 64, 4, 2, 1)

        # deco
        self.dcode1 = self.conv_resize(1536, 512, 3, 1, 1)
        self.dcode2 = self.conv_resize(1024, 256, 3, 1, 1)
        self.dcode3 = self.conv_resize(512, 128, 3, 1, 1)
        self.dcode4 = self.conv_resize(256, 128, 3, 1, 1)
        self.dcode5 = self.conv_resize(128, 128, 3, 1, 1)
        self.dcode6 = self.conv_resize(64, 64, 3, 1, 1)

        # last layer
        self.conv13 = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        x7 = self.dcode1(torch.cat([x5, self.conv7(x6)], 1))
        x8 = self.dcode2(torch.cat([x4, self.conv8(x7)], 1))
        x9 = self.dcode3(torch.cat([x3, self.conv9(x8)], 1))
        x10 = self.dcode4(torch.cat([x2, self.conv10(x9)], 1))
        x11 = self.dcode5(torch.cat([x1, self.conv11(x10)], 1))
        x12 = self.conv13(x11)
        x12 = nn.Upsample(scale_factor=2)(x12)
        t = nn.Tanh()
        x = t(x12)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu, use_cuda):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.use_cuda = use_cuda
        self.main = nn.Sequential(

            # 256
            nn.Conv2d(4, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 1, 4, 2, 1),
            nn.Sigmoid()

        )

    def forward(self, input):
        out = self.main(input)
        out = out.cuda() if self.use_cuda else out
        return out