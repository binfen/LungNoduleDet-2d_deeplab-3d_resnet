import torch
from torch.autograd import Variable
import sys
import numpy as np 

from utils import *


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()
    losses = AverageMeter()
    recalles = AverageMeter()
    accuracies = AverageMeter()

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

        loss = criterion(outputs, targets)
        recall = calculate_recall(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        recalles.update(recall, inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  loss=losses,
                  recall=recalles,
                  acc=accuracies))


    logger.log({'epoch': epoch, 'loss': losses.avg,'recall': recalles.avg,'acc': accuracies.avg})

    return losses.avg
