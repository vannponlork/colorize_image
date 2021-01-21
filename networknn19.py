from __future__ import print_function
import torch.nn as nn
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# follow https://arxiv.org/pdf/1803.05400.pdf

ngpu = 1
use_cuda = True if (torch.cuda.is_available() and ngpu > 0) else False
INPUT_DIM = 784
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


if torch.cuda.is_available():
    batch_size = 16
else:
    batch_size = 4


# class Generator(nn.Module):
#     def unet_conv_first(self, ch_in, ch_out, kernel, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
#             nn.LeakyReLU()
#         )
#
#
#     def unet_conv(self, ch_in, ch_out, kernel, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
#             nn.BatchNorm2d(ch_out),
#             nn.LeakyReLU()
#         )
#     def unet_conv_transpose(self, ch_in, ch_out, kernel, stride, padding):
#         return nn.Sequential(
#             nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU()
#         )
#
#     def conv_resize(self, ch_in, ch_out, kernel, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(),
#
#         )
#
#     def __init__(self, ngpu, use_cuda):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.use_cuda = use_cuda
#         self.conv1 = self.unet_conv_first(1, 64, 3, 1, 1)
#         self.conv2 = self.unet_conv(64, 128, 4, 2, 1)
#         self.conv3 = self.unet_conv(128, 256, 4, 2, 1)
#         self.conv4 = self.unet_conv(256, 256, 4, 2, 1)
#         self.conv5 = self.unet_conv(256, 512, 4, 2, 1)
#         self.conv6 = self.unet_conv(512, 512, 4, 2, 1)
#         self.conv7 = self.unet_conv(512, 512, 4, 2, 1)
#         self.conv8 = self.unet_conv(512, 512, 4, 2, 1)
#
#         self.conv9 = self.unet_conv_transpose(512, 512, 4, 2, 1)
#         self.conv10 = self.unet_conv_transpose(512, 512, 4, 2, 1)
#         self.conv11 = self.unet_conv_transpose(512, 512, 4, 2, 1)
#         self.conv12 = self.unet_conv_transpose(512, 256, 4, 2, 1)
#         self.conv13 = self.unet_conv_transpose(256, 256, 4, 2, 1)
#         self.conv14 = self.unet_conv_transpose(256, 128, 4, 2, 1)
#         self.conv15 = self.unet_conv_transpose(128, 64, 4, 2, 1)
#
#         # deco
#         self.dcode1 = self.conv_resize(1024, 512, 3, 1, 1)
#         self.dcode2 = self.conv_resize(1024, 512, 3, 1, 1)
#         self.dcode3 = self.conv_resize(1024, 512, 3, 1, 1)
#         self.dcode4 = self.conv_resize(512, 256, 3, 1, 1)
#         self.dcode5 = self.conv_resize(512, 256, 3, 1, 1)
#         self.dcode6 = self.conv_resize(256, 128, 3, 1, 1)
#         self.dcode7 = self.conv_resize(128, 64, 3, 1, 1)
#
#         # last layer
#         self.conv16 = nn.Conv2d(64, 2, 1, 1)
#
#     def forward(self, x):
#
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x7 = self.conv7(x6)
#         x8 = self.conv8(x7)
#
#         x9 = self.dcode1(torch.cat([x7, self.conv9(x8)], 1))
#         x10 = self.dcode2(torch.cat([x6, self.conv10(x9)], 1))
#         x11 = self.dcode3(torch.cat([x5, self.conv11(x10)], 1))
#         x12 = self.dcode4(torch.cat([x4, self.conv12(x11)], 1))
#         x13 = self.dcode5(torch.cat([x3, self.conv13(x12)], 1))
#         x14 = self.dcode6(torch.cat([x2, self.conv14(x13)], 1))
#         x15 = self.dcode7(torch.cat([x1, self.conv15(x14)], 1))
#         x16 = self.conv16(x15)
#         t = nn.Tanh()
#         x = t(x16)
#         return x

# class Discriminator(nn.Module):
#     def unet_conv_first(self, ch_in, ch_out, kernel, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
#             nn.LeakyReLU(0.2)
#         )
#
#     def unet_conv(self, ch_in, ch_out, kernel, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
#             nn.BatchNorm2d(ch_out),
#             nn.LeakyReLU(0.2)
#         )
#
#     def last_layer(self, ch_in, ch_out, kernel, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel, stride, padding)
#
#         )
#
#     def __init__(self, ngpu, use_cuda):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.use_cuda = use_cuda
#         # self.mbd1 = MinibatchDiscrimination(512, 512, batch_size)
#
#         self.conv1 = self.unet_conv_first(4, 64, 3, 1, 1)
#         self.conv2 = self.unet_conv(64, 128, 4, 2, 1)
#         self.conv3 = self.unet_conv(128, 256, 4, 2, 1)
#         self.conv4 = self.unet_conv(256, 256, 4, 2, 1)
#         self.conv5 = self.unet_conv(256, 512, 4, 2, 1)
#         self.conv6 = self.unet_conv(512, 512, 4, 2, 1)
#         self.conv7 = self.unet_conv(512, 512, 4, 2, 1)
#         self.conv8 = self.last_layer(512, 512, 4, 2, 1)
#
#         self.fc = nn.Linear(512 * 2 * 2, 1)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x7 = self.conv7(x6)
#         x8 = self.conv8(x7)
#
#         x9 = x8.view(-1, x8.shape[1] * x8.shape[2] * x8.shape[3]).to(device)
#         fc = nn.Linear(x8.shape[1] * x8.shape[2] * x8.shape[3], 1).to(device)
#         x12 = fc(x9).to(device)
#         t = nn.Sigmoid()
#         x = t(x12)
#         return x

class Generator(nn.Module):
    def unet_conv_first(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.LeakyReLU()
        )

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
    def upsampling(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

        )

    def __init__(self, ngpu, use_cuda):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.use_cuda = use_cuda
        self.conv1 = self.unet_conv_first(1, 64, 4, 2, 1)
        self.conv2 = self.unet_conv(64, 128, 4, 2, 1)
        self.conv3 = self.unet_conv(128, 256, 4, 2, 1)
        self.conv4 = self.unet_conv(256, 256, 4, 2, 1)
        self.conv5 = self.unet_conv(256, 512, 4, 2, 1)
        self.conv6 = self.unet_conv(512, 512, 4, 2, 1)
        self.conv7 = self.unet_conv(512, 512, 4, 2, 1)
        self.conv8 = self.unet_conv(512, 512, 4, 2, 1)

        self.conv9 = self.unet_conv_transpose(512, 512, 4, 2, 1)
        self.conv10 = self.unet_conv_transpose(512, 512, 4, 2, 1)
        self.conv11 = self.unet_conv_transpose(512, 256, 4, 2, 1)
        self.conv12 = self.unet_conv_transpose(512, 256, 4, 2, 1)
        self.conv13 = self.unet_conv_transpose(256, 128, 4, 2, 1)
        self.conv14 = self.unet_conv_transpose(256, 64, 4, 2, 1)
        self.conv15 = self.upsampling(128, 64, 4, 2, 1)

        # deco
        self.dcode1 = self.conv_resize(1024, 512, 3, 1, 1)
        self.dcode2 = self.conv_resize(1024, 512, 3, 1, 1)
        self.dcode3 = self.conv_resize(512, 512, 3, 1, 1)
        self.dcode4 = self.conv_resize(512, 256, 3, 1, 1)
        self.dcode5 = self.conv_resize(256, 256, 3, 1, 1)
        self.dcode6 = self.conv_resize(128, 128, 3, 1, 1)
        self.dcode7 = self.conv_resize(128, 64, 3, 1, 1)

        # last layer
        self.conv16 = nn.Conv2d(64, 2, 1, 1)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        xs = self.conv9(x7)
        x8 = self.dcode1(torch.cat([x6, self.conv9(x7)], 1))
        x9 = self.dcode2(torch.cat([x5, self.conv10(x8)], 1))
        x10 = self.dcode3(torch.cat([x4, self.conv11(x9)], 1))
        x11 = self.dcode4(torch.cat([x3, self.conv12(x10)], 1))
        x12 = self.dcode5(torch.cat([x2, self.conv13(x11)], 1))
        x13 = self.dcode6(torch.cat([x1, self.conv14(x12)], 1))
        x14 = self.conv15(x13)
        x15 = self.conv16(x14)
        # d = nn.Dropout()
        # m = nn.BatchNorm2d(2).to(device)
        # out = d(m(x15)).to(device)
        out = x15
        t = nn.Tanh()
        x = t(out).to(device)
        return x


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        mt = self.T.view(self.in_features, -1)
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class Discriminator(nn.Module):
    def unet_conv_first(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.LeakyReLU(0.2)
        )

    def unet_conv(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2)
        )

    def last_layer(self, ch_in, ch_out, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.LeakyReLU(0.2)

        )

    def __init__(self, ngpu, use_cuda):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.use_cuda = use_cuda
        self.mbd1 = MinibatchDiscrimination(512, 512, batch_size)

        self.conv1 = self.unet_conv_first(4, 64, 4, 2, 1)
        self.conv2 = self.unet_conv(64, 128, 4, 2, 1)
        self.conv3 = self.unet_conv(128, 256, 4, 2, 1)
        self.conv4 = self.unet_conv(256, 256, 4, 2, 1)
        self.conv5 = self.unet_conv(256, 512, 4, 2, 1)
        self.conv6 = self.unet_conv(512, 512, 4, 2, 1)
        self.conv7 = self.last_layer(512, 512, 4, 2, 1)



    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = x7.view(-1, 512).to(device)
        x9 = self.mbd1(x8).to(device)
        fc = nn.Linear(x9.shape[1], 1).to(device)
        x12 = fc(x9).to(device)
        t = nn.Sigmoid()
        x = t(x12)
        return x


