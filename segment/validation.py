import torch
from torch.autograd import Variable
import time
import sys
import numpy as np
from utils import *
import pdb
def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

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

        loss = criterion(outputs, labels.long())
        #loss = criterion(outputs[:,1], labels)
        acc = calculate_accuracy(outputs, labels)
        recall = calculate_recall(outputs, labels)
        redun = calculate_redundance(outputs, labels)

        losses.update(loss.data[0], inputs.size(0))
        recalles.update(recall, inputs.size(0))
        redunes.update(redun, inputs.size(0))
        batch_time.update(time.time() - end_time)

        end_time = time.time()

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


    logger.log({'epoch': epoch, 'loss': losses.avg,'recall':recalles.avg, 'redun':redunes.avg})

    return losses.avg
