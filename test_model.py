import torch
import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import DataLoader

import torchvision.utils
from torchvision.datasets import ImageFolder

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

    return video

def init_weights(m) :
    name = type(m)

    if name == nn.Conv3d or name == nn.ConvTranspose2d or name == nn.ConvTranspose3d :
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif name == nn.BatchNorm2d or name == nn.BatchNorm3d :
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#check GPU
is_gpu = torch.cuda.is_available()
print(is_gpu)

if is_gpu :
    dtype = torch.cuda.FloatTensor
else :
    dtype = torch.FloatTensor

D = discriminator().type(dtype)
gen_net, mask_net, static_net = generator()

gen_net = gen_net.type(dtype)
mask_net = mask_net.type(dtype)
static_net = static_net.type(dtype)

D.apply(init_weights)
gen_net.apply(init_weights)
mask_net.apply(init_weights)
static_net.apply(init_weights)

if is_gpu :
    real_labels = Variable(torch.ones(1).type(torch.cuda.LongTensor))
    fake_labels = Variable(torch.zeros(1).type(torch.cuda.LongTensor))
else :
    real_labels = Variable(torch.ones(1).type(torch.LongTensor))
    fake_labels = Variable(torch.zeros(1).type(torch.LongTensor))

criterion = nn.CrossEntropyLoss().type(dtype)
#criterion = nn.BCELoss().type(dtype)
#criterion = nn.BCEWithLogitsLoss().type(dtype)


d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(list(gen_net.parameters()) + list(mask_net.parameters()) + list(static_net.parameters()), lr=2e-4, betas=(0.5, 0.999))


data_transform = torchvision.transforms.Compose([torchvision.transforms.Scale((64,64)), torchvision.transforms.ToTensor()])
image_data = ImageFolder(root='./testset/', transform=data_transform)
data_loader = DataLoader(image_data, batch_size=1, shuffle=True)

video_data = None

for data, _ in data_loader:
    data.unsqueeze_(2)

    if torch.is_tensor(video_data):
        video_data = torch.cat((video_data, data), 2)
    elif video_data == None:
        video_data = data

video = Variable(video_data).type(dtype)

for epoch in range(1, 100) :

    # 1. Train Discriminator

    # 1-1. Real Video
    outputs = D(video).view(1, 2)
    d_loss_real = criterion(outputs.data, real_labels)

    # 1-2. Fake Video
    z = Variable(torch.randn(100) * 0.01).type(dtype)
    fake_videos = generate_video(gen_net, mask_net, static_net, z)
    outputs = D(fake_videos).view(1, 2)
    d_loss_fake = criterion(outputs, fake_labels)

    d_loss = d_loss_real + d_loss_fake

    D.zero_grad()
    d_loss.backward()
    d_optimizer.step()





    # 2. Train Generator
    z = Variable(torch.randn(100) * 0.01).type(dtype)
    fake_videos = generate_video(gen_net, mask_net, static_net, z)
    outputs = D(fake_videos).view(1, 2)

    g_loss = criterion(outputs, real_labels)

    D.zero_grad()
    gen_net.zero_grad()
    mask_net.zero_grad()
    static_net.zero_grad()

    g_loss.backward()
    g_optimizer.step()

    if epoch % 100 == 0 :
        print('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f' % (epoch, 1000, d_loss.data[0], g_loss.data[0]))



print('End Learning')

z = Variable(torch.randn(100) * 0.01).type(dtype)

for i in range(32) :
    fake_video = torch.squeeze(generate_video(gen_net, mask_net, static_net, z))[:,i,:,:]
    torchvision.utils.save_image(tensor=fake_video.data, filename="./test" + str(i+1) + ".png")