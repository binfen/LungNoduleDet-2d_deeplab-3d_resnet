# -*- encoding: utf-8 -*-
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from nodule import Nodule
from utils import Logger
from train import train_epoch
from validation import val_epoch
from test import test
import pdb
#pdb.set_trace()
if __name__ == '__main__':
    opt = parse_opts()
    opt.arch = '{}'.format(opt.model)
    if opt.root_path != '':
        opt.image_path = os.path.join(opt.root_path, opt.image_path)
        opt.label_path = os.path.join(opt.root_path, opt.label_path)
        opt.mask_path = os.path.join(opt.root_path, opt.mask_path)

        opt.train_file = os.path.join(opt.root_path, opt.train_file)
        opt.test_file = os.path.join(opt.root_path, opt.test_file)
        opt.val_file = os.path.join(opt.root_path, opt.val_file)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)+'-{}'.format(opt.arch)
        opt.savemodel_path = os.path.join(opt.result_path,opt.savemodel_path)

        if opt.resume:
            opt.resume_path = opt.savemodel_path
            opt.resume_file = os.path.join(opt.resume_path, opt.resume_file)

    print(opt)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    if not os.path.exists(opt.savemodel_path):
        os.makedirs(opt.savemodel_path)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt) #注意训练图像的通道数不同需要修改模型的输入
    print(model)
    
    a = np.array([0.01, 0.99], dtype=np.float32)
    w = torch.from_numpy(a).cuda()
    criterion = nn.CrossEntropyLoss(weight=w)
    #criterion = nn.NLLLoss(weight=w)

    if not opt.no_train:
        
        train_image = np.random.normal(size=(258,1,512,512))
        import pdb
        pdb.set_trace()
        train_label = np.random.randint(0,2,size=(258,512,512))
        train_image, train_label=torch.from_numpy(train_image),torch.from_numpy(train_label)
        training_data = torch.utils.data.TensorDataset(train_image, train_label)

        #training_data = Nodule(opt,opt.train_file)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)

        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'lr','recall','redun'])

        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'lr','recall','redun'])

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
        
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
        
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    if not opt.no_val:
        # val_image = np.random.normal(size=(1,1,512,512))
        # val_label = np.random.randint(0,2,size=(1,512,512))
        # val_image, val_label=torch.from_numpy(val_image),torch.from_numpy(val_label)
        # val_data = torch.utils.data.TensorDataset(val_image, val_label)
        val_data = Nodule(opt, opt.val_file)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=opt.train_batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss','recall','redun'])
    
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
            '''
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.learning_rate
            '''
    print('run')

    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        #training 
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)

        #validating
        if not opt.no_val:
            val_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
        
        if not opt.no_train and not opt.no_val:
            scheduler.step(val_loss)
    #testing
    if not opt.test:
        # test_image = np.random.normal(size=(1,1,512,512))
        # test_label = np.random.randint(0,2,size=(1,512,512))
        # test_image, test_label = torch.from_numpy(test_image), torch.from_numpy(test_label)
        # test_data = torch.utils.data.TensorDataset(test_image, test_label)
        test_data = Nodule(opt, opt.test_file)

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test(test_loader, model, opt)
    
