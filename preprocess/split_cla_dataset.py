# -*- encoding: utf-8 -*-
'''
@datatime: '18-6-6 下午7:30'

@author: wowjoy
'''
import os
from random import shuffle
main_ip = os.path.join(os.getcwd(),'../../datas/datasets/2D-deeplab-3D-resnet.pytorch/classify')
pos_ip = os.path.join(main_ip,'image_pos')
neg_ip = os.path.join(main_ip,'image_neg')

pos_names = os.listdir(pos_ip)
neg_names = os.listdir(neg_ip)

print('pos sample number:',len(pos_names),'neg sample number:',len(neg_names))

train_file = open(os.path.join(main_ip,'train_names.txt'),'w')
val_file = open(os.path.join(main_ip,'val_names.txt'),'w')
test_file = open(os.path.join(main_ip,'test_names.txt'),'w')

shuffle(pos_names)
shuffle(neg_names)
import pdb
#pdb.set_trace()
for i, name in enumerate(pos_names):
    if i < 20000:
        train_file.write('pos'+','+name+'\n')
    elif i < 30000:
        val_file.write('pos'+','+name+'\n')
    else:
        test_file.write('pos'+','+name+'\n') 

for j, name in enumerate(neg_names):

    if j < 100000:
        train_file.write('neg'+','+name+'\n')
    elif j < 15000:
        val_file.write('neg'+','+name+'\n')
    elif j < 151240:
        test_file.write('neg'+','+name+'\n') 
    else:
        break
    '''
    if j < 200000:
        train_file.write('neg'+','+name+'\n')
    elif j < 300000:
        val_file.write('neg'+','+name+'\n')
    else:
        test_file.write('neg'+','+name+'\n')     
    '''


train_file.close()
val_file.close()
test_file.close()




