# -*- encoding: utf-8 -*-
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import pdb
#pdb.set_trace()
from opts import parse_opts
from model import generate_model
from nodule import Nodule
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test

if __name__ == '__main__':
    opt = parse_opts()
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    if opt.root_path != '':
        opt.pos_image_path = os.path.join(opt.root_path, opt.pos_image_path)
        opt.neg_image_path = os.path.join(opt.root_path, opt.neg_image_path)

        opt.train_file = os.path.join(opt.root_path, opt.train_file)
        opt.val_file = os.path.join(opt.root_path, opt.val_file)
        opt.test_file = os.path.join(opt.root_path, opt.test_file)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)+'-{}'.format(opt.arch)

        opt.savemodel_path = os.path.join(opt.result_path, opt.savemodel_path)
        opt.pretrain_path = os.path.join(opt.result_path, opt.pretrain_path)

        if not os.path.exists(opt.savemodel_path):
            os.makedirs(opt.savemodel_path)
        if opt.resume:
            opt.resume_file = os.path.join(opt.savemodel_path, opt.resume_file)
        if opt.pretrain:
            opt.pretrain_file = os.path.join(opt.pretrain_path, opt.pretrain_file)
    print(opt)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    model, parameters = generate_model(opt)
    print(model)
    a = np.array([0.5, 0.5], dtype=np.float32)
    w = torch.from_numpy(a).cuda()
    #criterion = nn.CrossEntropyLoss(w)
    #criterion = nn.NLLLoss(weight=w)
    criterion = nn.NLLLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    
    if not opt.no_train:
        training_data = Nodule(opt.pos_image_path,opt.neg_image_path,opt.train_file)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'recall','acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'recall','acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
        '''

        '''
    
    if not opt.no_val:
        validation_data = Nodule(opt.pos_image_path,opt.neg_image_path,opt.val_file)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'recall','acc'])
    
    if opt.resume:
        print('loading checkpoint {}'.format(opt.resume_file))
        checkpoint = torch.load(opt.resume_file)
        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']  

        if not opt.no_cuda:   #使用GPU
            model.load_state_dict(state_dict)
        else:                 #使用CPU:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])



    print('run')
   
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        
        #training 
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        
        #validating
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
        
        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
    
    #testing
    if opt.test:
        test_data = Nodule(opt.pos_image_path,opt.neg_image_path,opt.test_file)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt)
    
