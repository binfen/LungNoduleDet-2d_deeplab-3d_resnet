#-*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json

from utils import *

def test(data_loader, model, opt):
    print('test')
    recalles = AverageMeter()
    accuracies = AverageMeter()
    model.eval()
    test_results = []
    f = open(os.path.join(opt.result_path, 'results.txt'),'w')
    f.write('name'+'\t'+'label'+'\t'+'prob'+'\t'+'pred'+'\n')

    for i, (names, targets) in enumerate(data_loader):
        inputs = np.zeros(shape=[len(names),1,opt.sample_size,opt.sample_size,opt.sample_size])
        for k, name in enumerate(names):
            inputs[k,:] = np.load(name)
        inputs = torch.from_numpy(inputs)

        if not opt.no_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True)

        inputs = Variable(inputs).type(torch.float32)        
        targets = Variable(targets).type(torch.long) 
        outputs = model(inputs)

        recall = calculate_recall(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        recalles.update(recall, inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs,dim=1)

        print('[{}/{}]\t'
              'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(i + 1,len(data_loader),recall=recalles,acc=accuracies))

        for j in range(outputs.size(0)):
            name = os.path.basename(names[j])
            label = targets[j]
            prob, pred = torch.topk(outputs[j], k=1)
            f.write(name+','+str(label.data.cpu().item())+','+str(prob.data.cpu().item())+','+str(pred.data.cpu().item())+'\n')

    f.close()
