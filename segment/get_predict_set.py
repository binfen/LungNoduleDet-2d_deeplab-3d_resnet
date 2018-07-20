# -*- encoding: utf-8 -*-
'''
@time: 2018  下午3:45

@author: wowjoy
'''

import os
import cv2
import numpy as np
import pandas as pd
from random import shuffle
import scipy.ndimage
import multiprocessing

def resize(data,size=32):
    image = np.zeros(shape=[size,size,size])
    height,width,depth = data.shape
    image[:height,:width,:depth]=data
    return image

def creat_sample(l):
    i, x, y, z, k = l
    data = np.load(os.path.join(data_ip, name + '.npy'))
    if data[y - k:y + k, x - k:x + k, z - k:z + k].shape == (size, size, size):
        nodule = data[y - k:y + k, x - k:x + k, z - k:z + k]
    else:
        nodule = resize(data[y - k:y + k, x - k:x + k, z - k:z + k],2*k)
    return nodule

def get_predict_sets(csv_file, seg_data, k = 16):
    nodule_list = []
    x = anno_info['voxelX'].values
    y = anno_info['voxelY'].values
    z = anno_info['voxelZ'].values
    #r = anno_info['voxelR'].values
    jobs = []
    for i in range(len(csv_file)):
        nodule = creat_sample(i,(int(x[i]),int(y[i]),int(z[i]),k)
        nodule_list.append(nodule)
    return nodule_list
    
if __name__ == '__main__':
    
    result = get_predict_sets()















