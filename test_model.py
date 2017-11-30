import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

def discriminator(input) :
    discriminator_model = nn.Sequential(
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

    scores = discriminator_model(input)

    return scores

def generator(z) :
    net_video = nn.Sequential(
        nn.ConvTranspose3d(in_channels = 100, out_channels = 512, kernel_size=(2,4,4)),
        nn.BatchNorm3d(num_features=512),
        nn.ReLU(inplace=True),

        nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.BatchNorm3d(num_features=256),
        nn.ReLU(inplace=True),

        nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.BatchNorm3d(num_features=128),
        nn.ReLU(inplace=True),

        nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.BatchNorm3d(num_features=64),
        nn.ReLU(inplace=True),
    )

    gen_net = nn.Sequential(
        net_video,
        nn.ConvTranspose3d(in_channels=64, out_channels=3, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.Tanh(),
    )

    mask_net = nn.Sequential(
        net_video,
        nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
        nn.Sigmoid(),
    )

    static_net = nn.Sequential(
        nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
        nn.Tanh(),
    )

    z_forward =  z.view(1, 100, 1, 1, 1)
    z_backword = z.view(1, 100, 1, 1)

    foreground = gen_net(z_forward)
    mask = mask_net(z_forward).expand(1, 3, 32, 64, 64)
    background = static_net(z_backword).view(1, 3, 1, 64, 64).expand(1, 3, 32, 64, 64)

    video = foreground * mask + background * (1 - mask)

    return video

dtype = torch.FloatTensor

z = torch.randn(1, 100, 1, 1, 1).type(dtype)
z_var = Variable(z.type(dtype))

fake_video = generator(z_var)
scores = discriminator(fake_video)

print(fake_video.size())
print(scores.size())