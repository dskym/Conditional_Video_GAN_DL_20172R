import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
import torchvision.transforms

from torch.utils.data import DataLoader

from torch.autograd import Variable

def discriminator() :
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

    return discriminator_model

def generator() :
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

    return gen_net, mask_net, static_net

def generate_video(gen_net, mask_net, static_net, z) :
    z_forward =  z.view(1, 100, 1, 1, 1)
    z_backword = z.view(1, 100, 1, 1)

    foreground = gen_net(z_forward)

    mask = mask_net(z_forward).expand(1, 3, 32, 64, 64)
    background = static_net(z_backword).view(1, 3, 1, 64, 64).expand(1, 3, 32, 64, 64)

    video = foreground * mask + background * (1 - mask)

data_transform = torchvision.transforms.Compose([torchvision.transforms.Scale((64,64)), torchvision.transforms.ToTensor()])
image_data = ImageFolder(root='./testset/', transform=data_transform)
data_loader = DataLoader(image_data, batch_size=1, shuffle=True)

video_data = None

for data, _ in data_loader :
    data.unsqueeze_(2)

    if torch.is_tensor(video_data):
        video_data = torch.cat((video_data, data), 2)
    elif video_data == None :
        video_data = data

print(video_data.shape)

dtype = torch.FloatTensor

video = Variable(video_data)

D = discriminator()
gen_net, mask_net, static_net = generator()

real_labels = Variable(torch.ones(1))
fake_labels = Variable(torch.zeros(1))

criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=3e-3,)
g_optimizer = torch.optim.Adam(gen_net.parameters(), lr=3e-3)




# 1. Train Discriminator

# 1-1. Real Video
outputs = D(video)
d_loss_real = criterion(outputs, real_labels)
real_score = outputs

# 1-2. Fake Video
z = Variable(torch.randn(100) * 0.01)
fake_videos = generate_video(gen_net, mask_net, static_net, z)
outputs = D(fake_videos)
d_loss_fake = criterion(outputs, fake_labels)
fake_score = outputs

d_loss = d_loss_real + d_loss_fake

D.zero_grad()
d_loss.backward()
d_optimizer.step()





# 2. Train Generator
z = Variable(torch.randn(100) * 0.01)
fake_videos = generate_video(gen_net, mask_net, static_net, z)
outputs = discriminator(fake_videos)

g_loss = criterion(outputs, real_labels)

D.zero_grad()
gen_net.zero_grad()
mask_net.zero_grad()
static_net.zero_grad()

g_loss.backward()
g_optimizer.step()