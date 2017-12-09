import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np

import re
import os

class VideoFolder(data.Dataset) :
    """
    /dataset/[class]/[video_num]/[image set - 32]
    """

    def __init__(self, root, transform=None) :
        self.video_path = []

        self.root = root
        self.transform = transform

        self.data_path = list(map(lambda x: os.path.join(self.root, x), os.listdir(self.root)))

        self.classes, self.class_to_idx = self.find_classes(self.root)

        for data_path in self.data_path :
            self.video_path = self.video_path + list(map(lambda x: os.path.join(data_path, x), os.listdir(data_path)))

    def find_classes(self, dir) :
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        self.num_classes = len(classes)
        
        return classes, class_to_idx

    def __getitem__(self, index) :
        video_path = self.video_path[index]
        image_path = list(map(lambda x: os.path.join(video_path, x), os.listdir(video_path)))

        cls = np.array([self.class_to_idx[str(os.path.dirname(video_path)).replace('./dataset/', '')]])
            
        tmp = torch.unsqueeze(torch.from_numpy(cls), 0)

        label =  torch.FloatTensor(1, self.num_classes)

        label.zero_()

        label.scatter_(1, tmp, 1)

        label = torch.squeeze(label, 0)

        video = None
        
        image_path.sort()

        for image in image_path :                
            image = Image.open(image).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            image.unsqueeze_(1)

            if torch.is_tensor(video):
                video = torch.cat((video, image), 1)
            elif video == None:
                video = image

        return (video, label)

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