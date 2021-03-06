#-*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from utils import *
from time import sleep

from threading import Thread
import matplotlib.pyplot as plt

def close(ts):
    sleep(ts)
    plt.close()

def predict(loader, model, opt):
    model.eval()
    preds = []
    for i, (inputs,) in enumerate(loader):
        inputs = Variable(inputs, requires_grad=False).type(torch.float32)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        pred = outputs.data.cpu().numpy()[:,1,:,:]
        preds.extend(pred)
    return preds
