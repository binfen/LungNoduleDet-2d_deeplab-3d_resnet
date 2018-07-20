# -*- encoding: utf-8 -*-
'''
@time: 2018  下午3:45

@author: wowjoy
'''

import os
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import pdb

main_ip = os.path.join(os.getcwd(),'../../datas/origin_datas/')
data_ip = os.path.join(main_ip,'image')
csv_ip = os.path.join(main_ip,'csv')

main_op = os.path.join(os.getcwd(), '../../datas/datasets/2D-deeplab-3D-resnet.pytorch/classify')
pos_data_op = os.path.join(main_op,'image_pos')
neg_data_op = os.path.join(main_op,'image_neg')

if not os.path.exists(pos_data_op):
    os.makedirs(pos_data_op)
if not os.path.exists(neg_data_op):
    os.makedirs(neg_data_op)

def padd_zeros(data,size=32):
    image = np.zeros(shape=[size,size,size])
    height,width,depth = data.shape
    image[:height,:width,:depth]=data
    return image

def soft_l1(img, a, b, w):
    img_o = img.copy()
    img_o = np.float32(img_o)
    if np.sum(img_o > b):
        x = img_o[img_o > b]
        img_o[img_o > b] = b + w / (1 + np.exp((b - x) / w)) - w / 2
    if np.sum(img_o < a):
        x = img_o[img_o < a]
        img_o[img_o < a] = a + w / (1 + np.exp((a - x) / w)) - w / 2
    img_o = img_o - (a - w / 2)
    img_o = img_o / (b - a + w) * 255
    return img_o.astype(np.uint8)

def create_singledata(data, x, y, z, size=32):
    r = size / 2
    if data[y - r:y + r, x - r:x + r, z - r:z + r].shape == (size, size, size):
        nodule = data[y - r:y + r, x - r:x + r, z - r:z + r]
    else:
        nodule = padd_zeros(data[y - r:y + r, x - r:x + r, z - r:z + r],size)
    nodule = soft_l1(nodule, -1350, 150, 150.0)
    return nodule

def get_pos_sample(pos_job):
    old_name,pos_num,x,y,z,size =pos_job
    data = np.load(os.path.join(data_ip, old_name))
    nodule = create_singledata(data, x, y, z, size)
    table = []
    # 数据增强：翻转
    table.append(nodule)
    table.append(nodule[::-1])
    table.append(nodule[:, ::-1])
    table.append(nodule[:, :, ::-1])
    table.append(nodule[::-1, ::-1])
    table.append(nodule[:, ::-1, ::-1])
    table.append(nodule[::-1, :, ::-1])
    table.append(nodule[::-1, ::-1, ::-1])
    for k, data in enumerate(table):
        new_name = '%06d.npy'%(8*pos_num+k+1)
        print('get pos sample, loop:', new_name)
        np.save(os.path.join(pos_data_op, new_name), data)

def get_classify_pos_set(pos_info,size):
    pool = ThreadPool(8)
    pos_jobs = []    
    pos_x = pos_info['voxelX'].values
    pos_y = pos_info['voxelY'].values
    pos_z = pos_info['voxelZ'].values
    pos_id = pos_info['ID'].values
    for j in range(pos_id.shape[0]):#
        old_name = pos_id[j]+'.npy'
        pos_jobs.append((old_name,j,int(pos_x[j]),int(pos_y[j]),int(pos_z[j]),size))
    pool.map(get_pos_sample,pos_jobs)
    pool.close() 
    pool.join()

def get_neg_sample(neg_job):
    old_name, neg_num, x, y, z, size =neg_job
    new_name = '%06d.npy' % neg_num
    print('get neg sample, loop:', new_name)
    data = np.load(os.path.join(data_ip, old_name))
    nodule = create_singledata(data, x, y, z, size)
    np.save(os.path.join(neg_data_op, new_name), nodule)

def get_classify_neg_set(neg_info,size):
    pool = ThreadPool(8)
    neg_jobs = []
    neg_x = neg_info['voxelX'].values
    neg_y = neg_info['voxelY'].values
    neg_z = neg_info['voxelZ'].values
    neg_id = neg_info['ID'].values
    for j in range(neg_id.shape[0]):#
        old_name = neg_id[j]+'.npy'
        neg_jobs.append((old_name,j,int(neg_x[j]),int(neg_y[j]),int(neg_z[j]),size))

    pool.map(get_neg_sample,neg_jobs)
    pool.close() 
    pool.join() 


def get_classify_set():
    size =32
    cand_file = os.path.join(csv_ip, 'candidates_detail_all.csv')
    anno_file = os.path.join(csv_ip, 'annotations_detail.csv')
    
    cand_info = pd.read_csv(cand_file,dtype={'ID':str})
    anno_info = pd.read_csv(anno_file,dtype={'ID':str})

    pos_info = anno_info.copy()
    neg_info = c_info[c_info['class']==0].iloc[:97200]
    get_classify_pos_set(pos_info,size)
    get_classify_neg_set(neg_info,size)
if __name__ == '__main__':
    get_classify_set()
















