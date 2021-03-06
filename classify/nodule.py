import torch
import torch.utils.data as data
import cv2
import os
from random import shuffle
import numpy as np


def make_dataset(pos_image_path,neg_image_path,train_file):
    pos_image_names = os.listdir(pos_image_path)
    neg_image_names = os.listdir(neg_image_path)

    names = [name.strip().split(',') for name in open(train_file,'r').readlines()]
    shuffle(names)
    dataset = []

    for i, (label,name) in enumerate(names):
        if label == 'pos':
            image_path = os.path.join(pos_image_path, name)
            if not os.path.exists(image_path):
                continue
            sample = {'image': image_path,
                      'label': 1}
        else:
            image_path = os.path.join(neg_image_path, name)
            if not os.path.exists(image_path):
                continue
            sample = {'image': image_path,
                      'label': 0}
        dataset.append(sample)
    return dataset

class Nodule(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, pos_image_path,neg_image_path,train_file):
        self.all_dataset = make_dataset(pos_image_path,neg_image_path,train_file)

    def loader(self,path):
        image  =np.load(path)
        return image    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        #path = self.all_dataset[index]['image']
        #image = self.loader(path)
        image = self.all_dataset[index]['image']
        label = self.all_dataset[index]['label']
        return image, label

    def __len__(self):
        return len(self.all_dataset)
