#-*- coding:utf-8 -*-
import os
import sys
import cv2
import json
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn
from utils import *
from opts import parse_opts
from multiprocessing import pool
import model
import segment
import classify
import pdb

def predict_demo():

    beg_time = time.time()
    opt = parse_opts()
    opt.reload_path = os.path.join(opt.root_path, opt.reload_path)
    #opt.image_path = os.path.join(opt.root_path,opt.image_path)
    #opt.mask_path = os.path.join(opt.root_path,opt.mask_path)

    opt.image_path = os.path.join(opt.root_path,'../../datas/test_data/'+opt.image_path)
    opt.mask_path = os.path.join(opt.root_path,'../../datas/test_data/'+opt.mask_path)

    '''#########################################################################'''
               # ----------reload pretrained models-------------#
    '''#########################################################################'''

    beg_time = time.time()
    # segment checkpoint
    seg_reload_file = os.path.join(opt.reload_path, opt.seg_reload_model)
    seg_model,_ = model.generate_model(opt, phase='segment')  
    seg_checkpoint = torch.load(seg_reload_file)   
    seg_state_dict = seg_checkpoint['state_dict']  
    if not opt.no_cuda:   #使用GPU
        seg_model.load_state_dict(seg_state_dict)
    else:                 #使用CPU:
        new_seg_state_dict = OrderedDict()
        for k, v in seg_state_dict.items():
            name = k[7:] # remove `module.`
            new_seg_state_dict[name] = v
        seg_model.load_state_dict(new_seg_state_dict)


    # loading classify checkpoint
    cla_reload_file = os.path.join(opt.reload_path, opt.cla_reload_model)
    cla_model,_ = model.generate_model(opt, phase='classify')  

    cla_checkpoint=torch.load(cla_reload_file)
    cla_state_dict = cla_checkpoint['state_dict']  
    if not opt.no_cuda:   #使用GPU
        cla_model.load_state_dict(cla_state_dict)
    else:                 #使用CPU:
        cla_new_state_dict = OrderedDict()
        for k, v in cla_state_dict.items():
            name = k[7:] # remove `module.`
            new_cla_state_dict[name] = v
        cla_model.load_state_dict(new_cla_state_dict)
    print('load model cost time:',time.time()-beg_time)


    # loop evalutation datas
    results = pd.DataFrame()
    names = os.listdir(opt.image_path)
    for i,name in enumerate(names):
        predict_beg_time = time.time()
        # load data
        image_path = os.path.join(opt.image_path,name)
        mask_path = os.path.join(opt.mask_path,name)
        state = get_patient_image_mask(image_path,mask_path, opt.seg_sample_size)
        if state == None:
            print('file have error shape')
            continue
        else:
            image, mask = state

        # seg
        seg_loader = get_seg_inputs(image, opt)
        seg_pred = segment.predict(seg_loader, seg_model, opt)
        seg_result,_ = get_seg_outputs(seg_pred, mask, opt.seg_pred_thresh)
        
        # cla
        cla_loader = get_cla_inputs(image, seg_result, opt)
        cla_prob,cla_pred = classify.predict(cla_loader, cla_model, opt)
        cla_result = get_cla_outpus(cla_prob,cla_pred,seg_result)

        # combin result
        ID=name.split('.')[0]
        cla_result['ID']=ID
        results=results.append(cla_result)
        print('loop %d: predict %s cost time:'%(i,name),time.time()-predict_beg_time)
    targets = pd.read_csv(os.path.join(opt.root_path,'../../datas/origin_datas/csv/annotations_detail.csv'),dtype={'ID':str})
    results.to_csv('outputs.csv',index=False)

    statistics(results,targets)
    print('predict all file cost time:',time.time()-beg_time)

if __name__ == '__main__':
    predict_demo()
    







