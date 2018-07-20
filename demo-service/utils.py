#-*-coding: utf-8 -*-
'''
@time: 2018.6.11 14:19
@author: wowjoy
'''

import os
import cv2
import time
import shutil
import numpy as np
import pandas as pd
from time import sleep
import SimpleITK as sitk
import torch.utils.data
from threading import Thread
import matplotlib.pyplot as plt
from multiprocessing import cpu_count,Pool,Process
from scipy.ndimage.interpolation import zoom
from sklearn.cluster import DBSCAN
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, remove_small_holes
from skimage.morphology import binary_dilation, binary_erosion, binary_closing

from boto.s3.key import Key
import boto.s3.connection
from boto.s3.connection import S3Connection
import os

########################################################################
region = ''
aws_access_key_id = ''
aws_secret_access_key = ''
bucket_name = ''

########################################################################

class S3(object):
    def __init__(self, bucket_name=bucket_name):

        self.conn = boto.s3.connect_to_region(
            region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            is_secure=True,
            calling_format=boto.s3.connection.OrdinaryCallingFormat()
        )
        self.bucket_name = bucket_name
        try:
            self.bucket = self.conn.get_bucket(self.bucket_name)
        except:
            raise ValueError('bucket:%s are not exist' % bucket_name)

    def upload_packetage(self, package_path):
        package_name = os.path.basename(package_path)
        package_key = Key(self.bucket, package_name)
        if package_key.exists():
            package_key.delete()
        else:
            packege_key.set_contents_from_filename(package_path)
        return

    def rename_package(self, package_old_name, package_new_name):
        package_old_key = Key(self.bucket, package_old_name)
        package_new_key = Key(self.bucket, package_new_name)
        if package_old_key.exists() and (not package_new_key.exists()):
            package_old_key.copy(self.bucket, package_new_key)
        if  package_new_key.exists():
            package_old_key.delete()
        return

    def download(self, name_list, save_path):
        if len(name_list) == 0:
            raise ValueError('package:%s are empty' % name_list)
        else:
            for i, name in enumerate(name_list) :
                package_key = Key(self.bucket, name)
                save_name = os.path.join(save_path,'%6d.jpg'%i)
                package_key.get_contents_to_filename(save_name)
        return

    def delete_packetage(self, package_name):
        package_key = Key(self.bucket, package_name)
        if package_key.exists():
            package_key.delete()
        else:
            raise ValueError('package:%s are not exist' % package_name)
        return

def Download_Dcm(name_list,save_path):
    s3=S3(bucket_name)
    s3.download(name_list, save_path)

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

def Resample(image, spacing, new_thickness=[1, 1, 1]):
    # Determine current pixel spacing
    resize_factor = spacing / new_thickness
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    #image.resize(new_shape)
    real_resize_factor = new_shape / image.shape
    image = zoom(image, real_resize_factor, mode='nearest').astype(np.float32)
    return image

def fix_contour_sclice(job):
    bi_mask, spacing=job
    spacing_y = spacing[0]
    spacing_x = spacing[1]
    disk_12 = disk(12, dtype=np.bool)
    bi_mask = binary_dilation(bi_mask, disk_12)
    bi_mask = remove_small_holes(bi_mask, min_size=3/spacing_x*3/spacing_y)
    return bi_mask


def get_mask(vol):
    '''
    method:获得肺腔掩码
    vol:CT体像素数据,三维
    '''
    if type(vol) is not np.ndarray:
        vol = np.array(vol)
    bi_vol = vol < -400
    pool=Pool(8)
    jobs=[]
    for i in range(vol.shape[0]):
        jobs.append((bi_vol[i, :, :]))
    bi_vol = pool.map(clear_border,jobs)
    pool.close()
    pool.join()
    bi_vol = np.array(bi_vol)
    label_vol = label(bi_vol, neighbors=8)
    regions = regionprops(label_vol)
    regions = [region for region in regions if region.bbox[0] < 300]#获得连通区面积小于300
    max_region = dict(index=0, area=0)
    bi_vol = np.zeros(bi_vol.shape, dtype=np.bool)
    for i, region in enumerate(regions):
        if region.area > max_region['area']:
            max_region['index'] = i
            max_region['area'] = region.area
    for region in regions:
        if region.area * 10 > max_region['area']:
            bi_vol = bi_vol + (label_vol == region.label)
    pool = Pool(8)  
    jobs=[]  
    for i in range(vol.shape[0]):
        jobs.append((1*bi_vol[i,:,:],(0.8,0.8,1)))
    bi_vol = pool.map(fix_contour_sclice,jobs)
    pool.close()
    pool.join()
    bi_vol = np.array(bi_vol)
    bi_vol = 1*(bi_vol > 0)
    return bi_vol

def get_patient_ct_data_mask_info_dcm(dcm_path,opt):

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    itk_img = reader.Execute()
    thickness = np.array(itk_img.GetSpacing())
    transform = np.array(itk_img.GetDirection())
    offset = np.array(itk_img.GetOrigin())
    shape = np.array(itk_img.GetSize())

    #获得image:[-2000,2000]
    old_array = sitk.GetArrayFromImage(itk_img)   # z, y, x
    if thickness[2]<=opt.thickness_z:
        new_array = old_array
    else:
        new_thickness = [thickness[0],thickness[1],opt.thickness_z]
        new_array = Resample(old_array, thickness, new_thickness=new_thickness)

    if new_array.shape[1]!=opt.seg_sample_size:
        new_array.resize((new_array.shape[0],opt.seg_sample_size,opt.seg_sample_size))

    #获得mask
    mask = get_mask(new_array)

    #获得data:[0,255]
    data = soft_l1(new_array, -1350, 150, 150.0)

    info=pd.DataFrame()
    # thickness是层厚矩阵
    info.loc[0,'thicknessX']=float(thickness[0])
    info.loc[0,'thicknessY']=float(thickness[1])
    info.loc[0,'thicknessZ']=float(thickness[2])
    info.loc[0,'thickness_z']=float(opt.thickness_z)

    # transform是旋转矩阵
    info.loc[0,'transformX']=np.around(float(transform[0]))
    info.loc[0,'transformY']=np.around(float(transform[4]))
    info.loc[0,'transformZ']=np.around(float(transform[8]))

    # offset是原点矩阵
    info.loc[0,'offsetX']=float(offset[0])
    info.loc[0,'offsetY']=float(offset[1])
    info['offsetZ']=float(offset[2])

    info.loc[0,'shapeX']=shape[0]
    info.loc[0,'shapeY']=shape[1]
    info.loc[0,'shapeZ']=shape[2]
    return new_array, data,mask,info


def get_patient_ct_data_mask_info_mhd(test_file,opt):
    itk_img = sitk.ReadImage(test_file)
    thickness = np.array(itk_img.GetSpacing())
    transform = np.array(itk_img.GetDirection())
    offset = np.array(itk_img.GetOrigin())
    shape = np.array(itk_img.GetSize())

    #获得image:[-2000,2000]
    old_array = sitk.GetArrayFromImage(itk_img)   # z, y, x
    if thickness[2]<=opt.thickness_z:
        new_array = old_array
    else:
        new_thickness = [thickness[0],thickness[1],opt.thickness_z]
        new_array = Resample(old_array, thickness, new_thickness=new_thickness)

    if new_array.shape[1]!=opt.seg_sample_size:
        new_array.resize((new_array.shape[0],opt.seg_sample_size,opt.seg_sample_size))

    #获得mask
    mask = get_mask(new_array)

    #获得data:[0,255]
    data = soft_l1(new_array, -1350, 150, 150.0)

    info=pd.DataFrame()
    # thickness是层厚矩阵
    info.loc[0,'thicknessX']=float(thickness[0])
    info.loc[0,'thicknessY']=float(thickness[1])
    info.loc[0,'thicknessZ']=float(thickness[2])
    info.loc[0,'thickness_z']=float(opt.thickness_z)

    # transform是旋转矩阵
    info.loc[0,'transformX']=np.around(float(transform[0]))
    info.loc[0,'transformY']=np.around(float(transform[4]))
    info.loc[0,'transformZ']=np.around(float(transform[8]))

    # offset是原点矩阵
    info.loc[0,'offsetX']=float(offset[0])
    info.loc[0,'offsetY']=float(offset[1])
    info['offsetZ']=float(offset[2])

    info.loc[0,'shapeX']=shape[0]
    info.loc[0,'shapeY']=shape[1]
    info.loc[0,'shapeZ']=shape[2]
    return new_array, data,mask,info


def get_seg_inputs(image, opt):
    image = np.expand_dims(image,axis=1)#四维
    seg_image = torch.from_numpy(image)
    seg_tensor = torch.utils.data.TensorDataset(seg_image) 

    seg_loader = torch.utils.data.DataLoader(
        seg_tensor,
        batch_size=opt.seg_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    return seg_loader

def get_seg_outputs(preds, mask, seg_pred_thresh):
    '''
    method: get segment result for classify
    preds: batch*height*width
    seg_outs: csv file, x,y,z,d_x,d_y, prob.最终的分割预测结果,用于分类
    '''
    
    #获得candidate 分割位置
    seg_results=pd.DataFrame()
    num=0
    for i, pred in enumerate(preds):
        _, contours, _ = cv2.findContours(255 * np.uint8(pred >= seg_pred_thresh), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for j, contour in enumerate(contours):
            # 计算轮廓的半径
            w_min, h_min, w, h = cv2.boundingRect(contour) 
            w_max, h_max = w_min+w, h_min+h
            prob = np.max(pred[h_min:h_max + 1, w_min:w_max + 1])
     
            x = np.int(np.ceil((w_max + w_min) / 2))
            y = np.int(np.ceil((h_max + h_min) / 2))
            z=  np.int(i)
            d = np.int(np.max([w, h]))

            if mask[z, y, x] == 0:
                pass
            else:
                seg_results.loc[num,'x']=x
                seg_results.loc[num,'y']=y
                seg_results.loc[num,'z']=z
                seg_results.loc[num,'dx']=w
                seg_results.loc[num,'dy']=h
                seg_results.loc[num,'prob']=prob
                num+=1

    #聚类获得最终分割结果
    seg_outs = pd.DataFrame(columns=seg_results.columns)
    data = seg_results[['x','y','z']]
    dbscan = DBSCAN(eps=6, min_samples=1)
    seg_results['cluster'] = dbscan.fit_predict(data)
    for j in seg_results.cluster.unique():
        voxel = seg_results.loc[seg_results.cluster==j,['x','y','z','dx','dy','prob']].values.mean(axis=0)

        seg_outs.loc[j,'x']=voxel[0]
        seg_outs.loc[j,'y']=voxel[1]
        seg_outs.loc[j,'z']=voxel[2]
        seg_outs.loc[j,'dx']=voxel[3]
        seg_outs.loc[j,'dy']=voxel[4]
        seg_outs.loc[j,'dz']=len(seg_results.loc[seg_results.cluster==j])
        seg_outs.loc[j,'seg_prob']=voxel[5]
 
    return seg_outs

def get_cla_sample(job):
    image, x, y, z, dx, dy, size = job
    if max(dx,dy) <=size+1:
        r = size / 2
        data = image[y - r:y + r, x - r:x + r, z - r:z + r].copy()#涉及浅拷贝问题:无法修改
        if data.shape != (size, size, size):
            data.resize((size,size,size))
    else:
        r = max(dx,dy)/2
        data = image[y - r:y + r, x - r:x + r, z - r:z + r].copy()
        if data.shape != (2*r,2*r,2*r):
            data.resize((2*r,2*r,2*r))
        data.resize((size,size,size))
    return data

def multhread_get_cla_inputs(image, seg_outs, opt):
    image= image.transpose(1,2,0)
    size = opt.cla_sample_size
    xs = seg_outs['x'].values
    ys = seg_outs['y'].values
    zs = seg_outs['z'].values
    dxs = seg_outs['dx'].values
    dys = seg_outs['dy'].values
    
    pool = Pool(8)
    jobs = []
    for i in range(len(xs)):
        jobs.append((image,int(xs[i]),int(ys[i]),int(zs[i]),int(dxs[i]),int(dys[i]), size))
    cla_samples = pool.map(get_cla_sample,jobs)
    pool.close()
    pool.join()

    cla_samples = np.array(cla_samples)
    cla_samples = np.expand_dims(cla_samples,axis=1)
    cla_samples = torch.from_numpy(cla_samples)
    cla_tensor = torch.utils.data.TensorDataset(cla_samples) 
    cla_loader = torch.utils.data.DataLoader(
        cla_tensor,
        batch_size=opt.cla_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    return cla_loader

def sigthread_get_cla_inputs(image, seg_outs, opt):
    image= image.transpose(1,2,0)
    size = opt.cla_sample_size
    xs = seg_outs['x'].values
    ys = seg_outs['y'].values
    zs = seg_outs['z'].values
    dxs = seg_outs['dx'].values
    dys = seg_outs['dy'].values
    cla_samples=[]
    for i in range(len(xs)):
        job=(image,int(xs[i]),int(ys[i]),int(zs[i]),int(dxs[i]),int(dys[i]), size)
        sample = get_cla_sample(job)
        cla_samples.append(sample)

    cla_samples = np.array(cla_samples)
    cla_samples = np.expand_dims(cla_samples,axis=1)
    cla_samples = torch.from_numpy(cla_samples)
    cla_tensor = torch.utils.data.TensorDataset(cla_samples) 
    cla_loader = torch.utils.data.DataLoader(
        cla_tensor,
        batch_size=opt.cla_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    return cla_loader

#TODO:获取分类结果
def get_cla_outpus(probs,preds,seg_results,info,ID):
    seg_results['ID']=ID
    seg_results['cla_prob']=probs
    seg_results['cla_pred']=preds
    info['ID']=ID
    cla_results = pd.merge(seg_results,info,on='ID')
    results  = cla_results[cla_results['cla_pred']==1]
    return results


def close(ts):
    sleep(ts)
    plt.close()
def show_predict_results(image, results,save_path):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for i in range(len(results)):
        x=int(results.iloc[i]['x'])
        y=int(results.iloc[i]['y'])
        z=int(results.iloc[i]['z'])
        dx=int(results.iloc[i]['dx'])
        dy=int(results.iloc[i]['dy'])
        r = max(dx,dy)+2
        x_min = x-r
        y_min = y-r
        x_max = x+r
        y_max = y+r

        ct_out = np.zeros(shape=(image.shape[1],image.shape[2],3))
        ct = image[z]
        ct_out[:, :, 1] = soft_l1(ct, -1200, 600, 600.0)
        ct_out[:, :, 0] = soft_l1(ct, -1200, -400, 600.0)
        ct_out[:, :, 2] = soft_l1(ct, -400, 600, 600.0)
        rec = cv2.rectangle(ct_out, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_path,'%d_%d.jpg'%(i,z)),rec)
        plt.subplot(121),plt.imshow(ct,'gray'),plt.title('%d ct'%i)
        plt.subplot(122),plt.imshow(rec,'gray'),plt.title('%d pred'%i)
        thread1 = Thread(target=close,args=(5,))
        thread1.start()
        plt.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

