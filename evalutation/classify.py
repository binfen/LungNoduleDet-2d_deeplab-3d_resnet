#-*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from utils import *
def predict(loader,model, opt):
    model.eval()
    probs=[]
    preds=[]
    for i, (inputs,) in enumerate(loader):
        inputs = Variable(inputs, requires_grad=False).type(torch.float32)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        prob, pred = torch.topk(outputs, k=1)
        probs.extend(prob.data.cpu().numpy()[:,0].tolist())
        preds.extend(pred.data.cpu().numpy()[:,0].tolist())
    return probs, preds
