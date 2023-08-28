from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import gaussian
import numpy as np
import torchvision as tv
import pandas as pd
from PIL import ImageFilter

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class Blur(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    def __call__(self, img):
        im2 = img.filter(ImageFilter.GaussianBlur(radius = self.kernel_size))
        return im2

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        if mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.RandomApply(transforms=[tv.transforms.RandomRotation((90, 90))], p=0.5),
                tv.transforms.RandomApply(transforms=[tv.transforms.RandomRotation((90, 90))], p=0.5),
                tv.transforms.RandomApply(transforms=[Blur(3)], p=0.5),
                tv.transforms.ColorJitter(0.5, 0.5),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        index = self.data.index[index]
        img_name = self.data['filename'][index]
        image = imread(img_name)
        image = gray2rgb(image)
        image = self._transform(image)

        return image, torch.tensor([self.data['crack'][index].astype('float32'), self.data['inactive'][index].astype('float32')])
