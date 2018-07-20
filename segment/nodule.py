#-*- coding:utf-8  -*-
import torch
import torch.utils.data as data
import cv2
import os
from random import shuffle
import numpy as np

def make_dataset(image_path,label_path,names_file,sample_size,in_channel):
    dataset=[]
    with open(names_file,'r') as f:
        names = [name.strip() for name in f.readlines()]
        for name in names:
            if not os.path.exists(os.path.join(image_path,name)):
                continue
            if name.split('.')[-1] == 'npy':
                image = np.load(os.path.join(image_path, name))
            else:
                if in_channel == 1:
                    image = cv2.imread(os.path.join(image_path, name), 0)
                elif in_channel == 3:
                    image = cv2.imread(os.path.join(image_path, name))

            if len(image.shape) == 2:   #灰度图像
                image = np.expand_dims(image,0)

            if not os.path.exists(os.path.join(label_path,name)):
                continue
            if name.split('.')[-1] == 'npy':
                label = np.load(os.path.join(label_path, name))
            else:
                label = cv2.imread(os.path.join(label_path, name),0)
                if np.max(label)>1:
                    label = label / np.max(label)
            if image.shape[2]>sample_size:
                print(name, image.shape)
                continue
            '''
            if image.shape[0]!=smaple_size:
                image = cv2.resize(image, sample_size)
                label = cv2.resize(label, sample_size)
            '''
            sample = {'image': image,
                      'label': label}
            dataset.append(sample)
    
    shuffle(dataset)
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

    def __init__(self, opt, names_file):
        self.image_path = opt.image_path
        self.label_path = opt.label_path
        self.names_file = names_file
        self.in_channel = opt.in_channel
        self.sample_size = opt.sample_size
        self.all_dataset = make_dataset(self.image_path, self.label_path, self.names_file,self.sample_size,self.in_channel)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label) where target is class_index of the target class.
        """
        image = self.all_dataset[index]['image']
        label = self.all_dataset[index]['label']
        return image, label

    def __len__(self):
        return len(self.all_dataset)
