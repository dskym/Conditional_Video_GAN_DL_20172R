import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        

        self.layer_video = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer_y = nn.Conv3d(in_channels=6, out_channels=32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        

        self.discriminator = nn.Sequential(

            nn.LeakyReLU(negative_slope=0.2, inplace=True),



            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.BatchNorm3d(num_features=128, eps=1e-03),

            nn.LeakyReLU(negative_slope=0.2, inplace=True),



            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.BatchNorm3d(num_features=256, eps=1e-03),

            nn.LeakyReLU(negative_slope=0.2, inplace=True),



            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),

            nn.BatchNorm3d(num_features=512, eps=1e-03),

            nn.LeakyReLU(negative_slope=0.2, inplace=True),



            nn.Conv3d(in_channels=512, out_channels=2, kernel_size=(2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0)),

        )



    

    def forward(self, video, y):

        out_video = self.layer_video(video)

        out_y = self.layer_y(y)

                             

        out_cat = torch.cat([out_video, out_y], 1)

                             

        out = self.discriminator(out_cat)

                             

        return out
