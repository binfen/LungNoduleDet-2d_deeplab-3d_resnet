import os,sys
import numpy as np
sys.path.append('*')
from datasets.nodule import Nodule
def get_training_set(opt):
    training_data = Nodule(opt.train_pos_image_path,opt.train_neg_image_path)
    return training_data


def get_validation_set(opt):
    validation_data = Nodule(opt.val_pos_image_path, opt.val_neg_image_path)
    return validation_data

def get_test_set(opt):
    test_data = Nodule(opt.test_pos_image_path,opt.test_neg_image_path)
    return test_data
