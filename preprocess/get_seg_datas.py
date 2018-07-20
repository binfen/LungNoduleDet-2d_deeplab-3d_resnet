# -*- encoding: utf-8 -*-
'''
@datatime: '18-6-6 下午2:25'

@author: wowjoy
'''

#TODO:重新修改函数进行模型训练
import os
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt


main_ip = os.path.join(os.getcwd(), '../../datas/origin_datas')
image_ip = os.path.join(main_ip, 'image')
label_ip = os.path.join(main_ip, 'label')
mask_ip = os.path.join(main_ip, 'mask')
csv_ip =os.path.join(main_ip, 'csv')

main_op = os.path.join(os.getcwd(), '../../datas/datasets/2D-deeplab-3D-resnet.pytorch/segment')
image_op = os.path.join(main_op, 'image')
label_op = os.path.join(main_op, 'label')
mask_op = os.path.join(main_op, 'mask')
if not os.path.exists(image_op):
    os.makedirs(image_op)
if not os.path.exists(label_op):
    os.makedirs(label_op)
if not os.path.exists(mask_op):
    os.makedirs(mask_op)

ch=1 #样本通道数


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


def resize(data, size):
    image = np.zeros(shape=[size, size])
    height, width = data.shape
    image[:height, :width] = data
    return image

def get_single_node_data(job):
    # id,name,x,y,z,rx,ry,rz,ch = job
    id, name, z, rz, ch = job
    # import pdb
    # pdb.set_trace()
    image = np.load(os.path.join(image_ip, name + '.npy'))
    height, width, depth = image.shape
    if height>512 and width>512:
        return 
    label = np.load(os.path.join(label_ip, name + '.npy'))

    z = int(z)
    # 设定分割选取的范围
    if rz < 2:
        R=0
    elif rz <= 10:
        R=3
    else:
        R=rz/2

    for k in range(int(max(z - R, 0)), int(min(z + R, depth))):
        # 加载相关原始数据
        print'{} {} {}'.format(id, name, k)
        sample_image = image[:, :, k]
        sample_label_out = 255 * label[:,:,k]
        sample_mask_out = 255 * mask[:,:,k]

        if ch == 1:
            sample_image_out = soft_l1(sample_image, -1350, 150, 150)

        elif ch == 3:
            sample_image_out = np.zeros((height, width, 3), dtype=np.float32)
            # 构造RGB三通道
            sample_image_out[:, :, 1] = soft_l1(sample_image, -1350, 150, 150)
            sample_image_out[:, :, 0] = soft_l1(sample_image, -1200, -400, 600.0)
            sample_image_out[:, :, 2] = soft_l1(sample_image, -400, 600, 600.0)

        # fig = plt.figure()
        cv2.imwrite(os.path.join(image_op, '{}_{}_{}.jpg'.format(id, name, k)), sample_image_out)
        cv2.imwrite(os.path.join(label_op, '{}_{}_{}.jpg'.format(id, name, k)), sample_label_out)
        #cv2.imwrite(os.path.join(mask_op, '{}_{}_{}.jpg'.format(id, name, k)), sample_mask_out)

        # plt.subplot(131), plt.imshow(sample_image_out, 'gray'), plt.title('{}_{}.jpg'.format(id, name))
        # plt.subplot(132), plt.imshow(sample_label_out, 'gray'), plt.title('{}_{}.jpg'.format(id, name))
        # plt.subplot(133), plt.imshow(sample_mask_out, 'gray'), plt.title('{}_{}.jpg'.format(id, name))
        # plt.show()
        # plt.close()
        # fig.clear()
    return


def main():
    #pool = ThreadPool(4)
    pool = Pool(8)
    csv_file = os.path.join(os.path.join(csv_ip, 'annotations_detail.csv'))
    df_node = pd.read_csv(csv_file, dtype={'ID': str})  # 指定列属性均为str类型

    names = df_node['ID'].values

    xs = (df_node['voxelX'].values).astype(np.int)
    ys = (df_node['voxelY'].values).astype(np.int)
    zs = (df_node['voxelZ'].values).astype(np.int)

    rxs = (df_node['r_X'].values).astype(np.int)
    rys = (df_node['r_Y'].values).astype(np.int)
    rzs = (df_node['r_Z'].values).astype(np.int)

    jobs = []
    for i in range(len(names)):
        job = (i, names[i], zs[i], rzs[i], ch)
        jobs.append(job)

    pool.map(get_single_node_data,jobs)
    pool.close()
    pool.join()
if __name__ == '__main__':
    main()
