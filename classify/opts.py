import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='../../datas/datasets/2D-deeplab-3D-resnet.pytorch/classify/',
        type=str,
        help='Root directory path of data')

    parser.add_argument(
        '--pos_image_path',
        default='image_pos',
        type=str,
        help='Directory path of positive image')

    parser.add_argument(
        '--neg_image_path',
        default='image_neg',
        type=str,
        help='Directory path of negative image')

    parser.add_argument(
        '--train_file',
        default='train_names.txt',
        type=str,
        help='Directory path of positive image')

    parser.add_argument(
        '--val_file',
        default='val_names.txt',
        type=str,
        help='Directory path of negative image')

    parser.add_argument(
        '--test_file',
        default='test_names.txt',
        type=str,
        help='Directory path of positive image')

    parser.add_argument(
        '--dataset',
        default='nodule',
        type=str,
        help='Used dataset (nodule)')

    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help=
        'Number of classes'
    )

    parser.add_argument(
        '--sample_size',
        default=32,
        type=int,
        help='Height, width and depth of inputs')

    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument(
        '--momentum', 
        default=0.9, 
        type=float, 
        help='Momentum')

    parser.add_argument(
        '--dampening', 
        default=0.9, 
        type=float, 
        help='dampening of SGD')

    parser.add_argument(
        '--weight_decay',
        default=1e-3, 
        type=float, 
        help='Weight Decay')

    parser.add_argument(
        '--nesterov', 
        action='store_true', 
        help='Nesterov momentum')
    parser.set_defaults(nesterov=False)

    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently only support SGD')

    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )

    parser.add_argument(
        '--batch_size', 
        default=64, 
        type=int,  
        help='Batch Size')

    parser.add_argument(
        '--n_epochs',
        default=10,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )

    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')

    parser.add_argument(
        '--resume',
        default=False,
        type=bool,
        help='wether to use chevkpoint')

    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='')

    parser.add_argument(
        '--resume_file',
        default='save_10.pth',
        type=str,
        help='')

    parser.add_argument(
        '--pretrain', 
        default=False, 
        type=bool, 
        help='wether to use Pretrained model')

    parser.add_argument(
        '--pretrain_path', 
        default='save_models', 
        type=str, 
        help='Pretrained model (.pth)')

    parser.add_argument(
        '--pretrain_file', 
        default='save_10.pth', 
        type=str, 
        help='Pretrained model (.pth)')

    parser.add_argument(
        '--savemodel_path',
        default='save_models',
        type=str,
        help='Save trained data (.pth)')

    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)

    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)

    parser.add_argument(
        '--test',
        action='store_true',
        help='If true, test is performed.')
    parser.set_defaults(test=True)

    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)

    parser.add_argument(
        '--no_cuda', action='store_true', 
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument(
        '--n_threads',
        default=8,
        type=int,
        help='Number of threads for multi-thread loading')

    parser.add_argument(
        '--sample_duration',
        default=16,
        type=int,
        help='Temporal duration of inputs')

    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')

    parser.add_argument(
        '--model',
        default='densenet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')

    parser.add_argument(
        '--model_depth',
        default=121,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')

    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')

    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, 
        help='Wide resnet k')

    parser.add_argument(
        '--resnext_cardinality',
        default=128,
        type=int,
        help='ResNeXt cardinality')

    parser.add_argument(
        '--manual_seed', default=1, type=int, 
        help='Manually set random seed')

    args = parser.parse_args()

    return args
