#-*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
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
        self.log_file = open(path, 'w')
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

def calculate_iou(pred, target):
    '''calculate IOU'''
    p = pred.data.max(1)[1].view(pred.size(0), -1).float()
    q = target.data.view(target.size(0), -1).float()
    a = (p * q).sum(1)
    b = (p.sum(1) + q.sum(1) - (p * q).sum(1))
    iou = (a / b).mean()
    return iou

def calculate_redundance(pred, target):
    ''''calculate false smaple'''
    p = pred.data.max(1)[1].view(pred.size(0), -1).float()
    q = target.data.view(target.size(0), -1).float()
    redun = (p.sum(1) - (p * q).sum(1)).mean()
    return redun

def calculate_recall(pred, target):
    ''''calculate recall'''
    p = pred.data.max(1)[1].view(pred.size(0), -1).float()
    q = target.data.view(target.size(0), -1).float()
    a = (p * q).sum(1)
    b = q.sum(1)
    recall = (a / (b+1)).mean()
    return recall

def calculate_precision(pred, target):
    '''calculate accuracy'''
    p = pred.data.max(1)[1].view(pred.size(0), -1).float()
    q = target.data.view(pred.size(0), -1).float()
    a = (p * q).sum(1)
    b = p.sum(1) + 1
    precision = (a / b).mean()
    return precision

def calculate_accuracy(pred, target):
    pred = pred.long()
    target = target.long()
    p = pred.data.max(1)[1].view(pred.size(0),-1)
    q = target.data.view(pred.size(0),-1)
    correct = p.eq(q)
    n_correct_elems = correct.float().sum().data[0]
    return n_correct_elems / (p.size(0)*p.size(1))



