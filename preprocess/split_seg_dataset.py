# -*- encoding: utf-8 -*-
'''
@datatime: '18-6-6 下午7:30'

@author: wowjoy
'''
import os
from random import shuffle
main_ip = os.path.join(os.getcwd(),'../../datas/datasets/2D-deeplab-3D-resnet.pytorch/segment')
data_ip = os.path.join(main_ip,'image')

names = os.listdir(data_ip)
print len(names)

train_file = open(os.path.join(main_ip,'train_names.txt'),'w')
val_file = open(os.path.join(main_ip,'val_names.txt'),'w')
test_file = open(os.path.join(main_ip,'test_names.txt'),'w')

shuffle(names)
import pdb
#pdb.set_trace()
for i, name in enumerate(names):
    if i < 11000:
        train_file.write(name+'\n')
    elif i < 11180:
        val_file.write(name+'\n')
    else:
        test_file.write(name+'\n')

train_file.close()
val_file.close()




