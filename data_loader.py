import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

import os

class VideoFolder(data.Dataset) :
    """
    /dataset/[class]/[video_num]/[image set - 32]
    """

    def __init__(self, root, transform=None) :
        self.video_path = []

        self.data_path = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

        for data_path in self.data_path :
            self.video_path = self.video_path + list(map(lambda x: os.path.join(data_path, x), os.listdir(data_path)))

        self.transform = transform

    def __getitem__(self, index) :
        print(index)
        video_path = self.video_path[index]
        image_path = list(map(lambda x: os.path.join(video_path, x), os.listdir(video_path)))

        video = None

        for i, (image) in enumerate(image_path) :
            if i == 32 :
                break
            image = Image.open(image).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            image.unsqueeze_(0)

            if torch.is_tensor(video):
                video = torch.cat((video, image), 0)
            elif video == None:
                video = image

        return video

    def __len__(self) :
        return len(self.video_path)

def get_loader(data_path, image_size, batch_size, num_workers=2):
    transform = transforms.Compose([
        transforms.Scale((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = VideoFolder(data_path, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    return data_loader

if __name__ == '__main__':
    data_loader = get_loader(data_path='./dataset', image_size=64, batch_size=1, num_workers=2)

    for video in data_loader :
        pass