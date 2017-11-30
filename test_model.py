import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        #print(N, C, H, W)
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def discriminator() :
    dtype = torch.FloatTensor

    model = nn.Sequential(
        nn.Conv3d (in_channels=3, out_channels=64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.BatchNorm3d(num_features=128, eps=1e-03, momentum=0.1, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.BatchNorm3d(num_features=256, eps=1e-03, momentum=0.1, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.BatchNorm3d(num_features=512, eps=1e-03, momentum=0.1, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv3d(in_channels=512, out_channels=2, kernel_size=(2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0)),
    )

    x = torch.randn(1, 3, 32, 64, 64).type(dtype)
    x_var = Variable(x.type(dtype))

    s = model(x_var)

    print(s)

