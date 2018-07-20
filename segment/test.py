#-*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
from time import sleep
from threading import Thread
import matplotlib.pyplot as plt
from utils import AverageMeter
import pdb

def close(ts):
    sleep(ts)
    plt.close()

def test(data_loader, model, opt):
    print('test')

    model.eval()
    preds = []
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    for i, (names, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda(async=True)

        inputs = Variable(inputs).type(torch.float)
        labels = Variable(labels).type(torch.float)

        outputs = model(inputs)
        '''
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)
        for j in range(outputs.size(0)):
            pred = outputs.data.max(1)[1].data.cpu().numpy()[j]
            label = labels.data.numpy()[j]            
            plt.subplot(121),plt.imshow(pred,'gray'),plt.title('%d pred'%j)
            plt.subplot(122),plt.imshow(255*label,'gray'),plt.title('%d label'%j)
            thread1 = Thread(target=close,args=(3,))
            thread1.start()
            plt.show()
        '''
        outputs = F.softmax(outputs)
        pred = outputs.data.cpu().numpy()[:,1,:,:]
        preds.extend(pred)
    return preds

