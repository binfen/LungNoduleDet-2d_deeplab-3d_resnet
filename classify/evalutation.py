#-*- coding:utf-8 -*-

import os
import cv2
import sys
import json
import time
import torch
from models import *
from utils import *
import numpy as np
import pandas as pd
import scipy.ndimage
from random import shuffle
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from multiprocessing import Pool
from torch.autograd import Variable
main_ip = '../../origin_datas'
data_ip = os.path.join(main_ip,'image')
csv_ip =  os.path.join(main_ip,'csv')
anno_file = os.path.join(csv_ip,'annotations_detail.csv')
cand_file = os.path.join(csv_ip,'candidates_detail.csv')
resume_file='../data/classify/results-resnet-50/save_models/save_10.pth'

def padd_zero(data,size=32):
    image = np.zeros(shape=[size,size,size])
    height,width,depth = data.shape
    image[:height,:width,:depth]=data
    return image

def creat_sample(l):
    name, x, y, z, size = l
    k = size/2
    image = np.load(os.path.join(data_ip, name))
    data = image[y - k:y + k, x - k:x + k, z - k:z + k].copy()
    if data.shape != (size, size, size):
        data = padd_zero(data,size)
    data = np.expand_dims(data,axis=0)
    print(name)
    return data

def get_evl_sets():
    anno_info = pd.read_csv(anno_file,dtype={'ID':str})[['ID','voxelX','voxelY','voxelZ']]
    anno_info['calss']=1
    cand_info = pd.read_csv(cand_file,dtype={'ID':str})[['ID','voxelX','voxelY','voxelZ','class']]
    
    anno_info = anno_info.iloc[:32]
    cand_info = cand_info.iloc[:32]

    anno_info = pd.concat([anno_info,cand_info],ignore_index=True)

    pool=Pool(8)
    size=32
    ids= anno_info['ID'].values
    xs = anno_info['voxelX'].values
    ys = anno_info['voxelY'].values
    zs = anno_info['voxelZ'].values
    jobs = []
    for i, name in enumerate(ids):
        name = name + '.npy'
        jobs.append((name,int(xs[i]),int(ys[i]),int(zs[i]),size))
    results = pool.map(creat_sample, jobs)
    pool.close()
    pool.join()
    results = np.array(results)
    results = torch.from_numpy(results)

    data_tensor = torch.utils.data.TensorDataset(results) 
    data_loader = torch.utils.data.DataLoader(
        data_tensor,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True)
    return data_loader

data_loader = get_evl_sets()



model = resnet.resnet50(num_classes=2,shortcut_type='B')
model = model.cuda()
model = nn.DataParallel(model, device_ids=None)

parameters = model.parameters

print('loading checkpoint')
checkpoint = torch.load(resume_file)
state_dict = checkpoint['state_dict']  
model.load_state_dict(state_dict)

def predict(data_loader, model):
    print('predict')
    model.eval()
    probs = []
    preds = []

    for i, (inputs,) in enumerate(data_loader):
        inputs = Variable(inputs, volatile=True).type(torch.float32)

        outputs = model(inputs)
        outputs = F.softmax(outputs)
        prob, pred = torch.topk(outputs, k=1)
        probs.append(prob)
        preds.append(pred)
    
    import pdb
    pdb.set_trace()
predict(data_loader, model)
