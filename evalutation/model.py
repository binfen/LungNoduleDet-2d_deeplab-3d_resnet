#-*- coding:utf-8 -*-
import torch
from torch import nn
import os,sys
sys.path.append(os.path.abspath('*'))
from models import deeplab,resnet, pre_act_resnet, wide_resnet, resnext, densenet
def generate_model(opt,phase):
    if phase =='segment':
        assert opt.seg_model in ['deeplab']
        if opt.seg_model == 'deeplab':
            model = deeplab.Net(in_channel=opt.in_channel, num_classes=opt.n_classes)
    elif phase=='classify':
        assert opt.cla_model in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

        if opt.cla_model == 'resnet':
            assert opt.cla_model_depth in [10, 18, 34, 50, 101, 152, 200]

            from models.resnet import get_fine_tuning_parameters
 
            if opt.cla_model_depth == 10:
                model = resnet.resnet10(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 18:
                model = resnet.resnet18(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 34:
                model = resnet.resnet34(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 50:
                model = resnet.resnet50(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 101:
                model = resnet.resnet101(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 152:
                model = resnet.resnet152(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 200:
                model = resnet.resnet200(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
        elif opt.cla_model == 'wideresnet':
            assert opt.cla_model_depth in [50]

            from models.wide_resnet import get_fine_tuning_parameters

            if opt.cla_model_depth == 50:
                model = wide_resnet.resnet50(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut,
                    k=opt.wide_resnet_k)
        elif opt.cla_model == 'resnext':
            assert opt.cla_model_depth in [50, 101, 152]

            from models.resnext import get_fine_tuning_parameters

            if opt.cla_model_depth == 50:
                model = resnext.resnet50(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut,
                    cardinality=opt.resnext_cardinality)
            elif opt.cla_model_depth == 101:
                model = resnext.resnet101(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut,
                    cardinality=opt.resnext_cardinality)
            elif opt.cla_model_depth == 152:
                model = resnext.resnet152(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut,
                    cardinality=opt.resnext_cardinality)
        elif opt.cla_model == 'preresnet':
            assert opt.cla_model_depth in [18, 34, 50, 101, 152, 200]

            from models.pre_act_resnet import get_fine_tuning_parameters

            if opt.cla_model_depth == 18:
                model = pre_act_resnet.resnet18(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 34:
                model = pre_act_resnet.resnet34(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 50:
                model = pre_act_resnet.resnet50(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 101:
                model = pre_act_resnet.resnet101(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 152:
                model = pre_act_resnet.resnet152(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
            elif opt.cla_model_depth == 200:
                model = pre_act_resnet.resnet200(
                    num_classes=opt.n_classes,
                    shortcut_type=opt.cla_resnet_shortcut)
        elif opt.cla_model == 'densenet':
            assert opt.cla_model_depth in [121, 169, 201, 264]
    
            from models.densenet import get_fine_tuning_parameters

            if opt.cla_model_depth == 121:
                model = densenet.densenet121(
                    num_classes=opt.n_classes)
            elif opt.cla_model_depth == 169:
                model = densenet.densenet169(
                    num_classes=opt.n_classes)
            elif opt.cla_model_depth == 201:
                model = densenet.densenet201(
                    num_classes=opt.n_classes)
            elif opt.cla_model_depth == 264:
                model = densenet.densenet264(
                    num_classes=opt.n_classes)
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
    return model, model.parameters()
