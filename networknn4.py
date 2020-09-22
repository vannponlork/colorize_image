from __future__ import print_function
import torch.nn as nn
import torch

# follow https://arxiv.org/pdf/1803.05400.pdf

ngpu = 1
use_cuda = True if (torch.cuda.is_available() and ngpu > 0) else False
# class Generator(nn.Module):
#     def __init__(self, ngpu, use_cuda):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.use_cuda = use_cuda
#         self.main = nn.Sequential(
#
#             nn.Conv2d(1, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2),
#             # 32
#             nn.Conv2d(128, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2),
#             #
#             # # # 16
#             nn.Conv2d(256, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2),
#
#             # # 8
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2),
#             #
#             # # 4
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2),
#
#             # # 2
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.Upsample(scale_factor=2),
#             nn.ReLU(),
#
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#
#             nn.Upsample(scale_factor=2),
#             nn.ReLU(),
#
#             # 8
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#
#             # 16
#             nn.Upsample(scale_factor=2),
#             nn.ReLU(),
#
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#
#             # 32
#             nn.Upsample(scale_factor=2),
#             nn.ReLU(),
#
#             nn.Conv2d(512, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#
#             # 64
#             nn.Upsample(scale_factor=2),
#             nn.ReLU(),
#
#             nn.Conv2d(256, 128, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#
#             # 128
#             nn.Upsample(scale_factor=2),
#             nn.ReLU(),
#
#             nn.Conv2d(128, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#             # 256
#
#             nn.Upsample(scale_factor=2),
#             nn.ReLU(),
#
#             nn.Conv2d(64, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#
#             nn.Conv2d(64, 2, 3, 1, 1, bias=True),
#             nn.Tanh(),
#         )
#
#     def forward(self, input):
#         out = self.main(input)
#         out = out.cuda() if self.use_cuda else out
#         return out
# class Discriminator(nn.Module):
#     def __init__(self, ngpu, use_cuda):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.use_cuda = use_cuda
#         self.main = nn.Sequential(
#
#             nn.Conv2d(3, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(64, 128, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 128, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#
#             # 64
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(128, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             # 32
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             # 16
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#
#             # 8
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(256, 256, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             # 4
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             # 2
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, 1, 1, bias=True),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),
#
#             nn.Conv2d(512, 1, 1, 2, 0, bias=True),
#             nn.Sigmoid()
#
#         )
#
#     def forward(self, input):
#         out = self.main(input)
#         out = out.cuda() if self.use_cuda else out
#         return out


class Generator(nn.Module):
    def __init__(self, ngpu, use_cuda):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.use_cuda = use_cuda
        self.main = nn.Sequential(

            nn.Conv2d(1, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            # 32
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            #
            # # # 16
            nn.Conv2d(256, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, 3, 1, 1, bias=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=True),

            nn.Upsample(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(1024, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(512, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, 1, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, input):
        out = self.main(input)
        out = out.cuda() if self.use_cuda else out
        return out

class Discriminator(nn.Module):
    def __init__(self, ngpu, use_cuda):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.use_cuda = use_cuda
        self.main = nn.Sequential(

            nn.Conv2d(3, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 64
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 32
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 16
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 8
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 4
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 2
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 1, 2, 0, bias=True),
            nn.Sigmoid()

        )

    def forward(self, input):
        out = self.main(input)
        out = out.cuda() if self.use_cuda else out
        return out