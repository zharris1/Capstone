'''
This is the data that will serve as the data that the stylist will read in

Will use: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''
import os

import pandas as pd

import torch 
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class ConsolidatedData(Dataset):

    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

        '''
        Not sure if we really need both __training_data__ and __test_data__; maybe it's best to have one loader and separate the data at a later stage?
        '''

    def __training_data__(self):
        training_data = datasets.FashionMNIST(root = self.img_dir, train = True, download = True, transform = self.transform)
        return training_data

    def __test_data__(self):
        test_data = datasets.FashionMNIST(root = self.img_dir, train = False, download = True, transform = self.transform)
        return test_data

    def dataloader(self, batch_size):
        train_dataloader = DataLoader(self.__training_data__, batch_size, shuffle = True)
        test_dataloader = DataLoader(self.__test_data__, batch_size, shuffle = True)
        return train_dataloader, test_dataloader