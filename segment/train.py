import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np

from utils import * 
import pdb

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    recalles = AverageMeter()
    redunes = AverageMeter()

    end_time = time.time()

    for i, (inputs, labels) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda(async=True)

        inputs = Variable(inputs).type(torch.float)        
        labels = Variable(labels).type(torch.float)        
        outputs = model(inputs)

        #binary_cross_entropy loss use type of tensor is float.
        #loss = criterion(outputs[:,1], labels)
        loss = criterion(outputs, labels.long())
        recall = calculate_recall(outputs, labels)
        redun = calculate_redundance(outputs, labels)

        losses.update(loss.data[0], inputs.size(0))     
        recalles.update(recall, inputs.size(0))
        redunes.update(redun,inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'lr': optimizer.param_groups[0]['lr'],
            'recall': recalles.val,
            'redun':redunes.val
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
              'Redun {redun.val:.3f} ({redun.avg:.3f})\t'.format(epoch,
                                                                 i + 1,
                                                                 len(data_loader),
                                                                 batch_time=batch_time,
                                                                 data_time=data_time,
                                                                 loss=losses,
                                                                 recall=recalles,
                                                                 redun=redunes))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[0]['lr'],
        'recall':recalles.avg,
        'redun':redunes.avg
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.savemodel_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
