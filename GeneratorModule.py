import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        

        self.layer_3d_video = nn.ConvTranspose3d(in_channels=100, out_channels=256, kernel_size=(2,4,4))

        self.layer_2d_video = nn.ConvTranspose2d(in_channels=100, out_channels=256, kernel_size=4, stride=1, padding=0)



        self.layer_3d_y = nn.ConvTranspose3d(in_channels=6, out_channels=256, kernel_size=(2,4,4))        

        self.layer_2d_y = nn.ConvTranspose2d(in_channels=6, out_channels=256, kernel_size=4, stride=1, padding=0)

        

        self.net_video = nn.Sequential(

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

            nn.ReLU(inplace=True)

        )



        self.gen_net = nn.Sequential(

            nn.ConvTranspose3d(in_channels=64, out_channels=3, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.Tanh()

        )



        self.mask_net = nn.Sequential(

            nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.Sigmoid()

        )



        self.static_net = nn.Sequential(

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

            nn.Tanh()

        )



    def forward(self, z, y):

        

        local_batch_size = z.size()[0]

        

        z_forgeround =  z.view(-1, 100, 1, 1, 1)

        z_background = z.view(-1, 100, 1, 1)

        

        y_foreground =  y.view(-1, 6, 1, 1, 1)

        y_background = y.view(-1, 6, 1, 1)

        

        out_3d_video = self.layer_3d_video(z_forgeround)

        out_2d_video = self.layer_2d_video(z_background)

        

        out_3d_y = self.layer_3d_y(y_foreground)

        out_2d_y = self.layer_2d_y(y_background)



        out_cat_3d = torch.cat([out_3d_video, out_3d_y],1)

        out_cat_2d = torch.cat([out_2d_video, out_2d_y],1)

        

        m_net_video = self.net_video(out_cat_3d)

        

        m_gen_net = self.gen_net(m_net_video)

        m_mask_net = self.mask_net(m_net_video)

        

        m_static_net = self.static_net(out_cat_2d)

        

        foreground = m_gen_net



        mask = m_mask_net.expand(local_batch_size, 3, 32, 64, 64)



        background = m_static_net.view(local_batch_size, 3, 1, 64, 64).expand(local_batch_size, 3, 32, 64, 64)

        

        video = foreground * mask + background * (1 - mask)



        return video
