import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='',
        type=str,
        help='Root directory path of data')

    parser.add_argument(
        '--reload_path',
        default='trained_models',
        type=str,
        help='')

    parser.add_argument(
        '--seg_reload_model',
        default='seg_save_158.pth',
        type=str,
        help='')

    parser.add_argument(
        '--cla_reload_model',
        default='cla_save_15.pth',
        type=str,
        help='')

    parser.add_argument(
        '--data_path',
        default='../../datas/demo_datas',
        type=str,
        help='')

    parser.add_argument(
        '--outputs_path',
        default='outputs',
        type=str,
        help='')


    parser.add_argument(
        '--phase',
        default=True,
        type=bool,
        help='')

    parser.add_argument(
        '--in_channel',
        default=1,
        type=int,
        help='Directory path of positive image')

    parser.add_argument(
        '--thickness_z',
        default=1,
        type=float,
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
        '--seg_sample_size',
        default=512,
        type=int,
        help='Height, width and depth of inputs')

    parser.add_argument(
        '--seg_pred_thresh',
        default=0.7,
        type=float,
        help='thresh')

    parser.add_argument(
        '--cla_sample_size',
        default=32,
        type=int,
        help='Height, width and depth of inputs')

    parser.add_argument(
        '--cla_resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')

    parser.add_argument(
        '--seg_batch_size', 
        default=8,
        type=int,  
        help='Batch Size')

    parser.add_argument(
        '--cla_batch_size', 
        default=32,
        type=int,  
        help='Batch Size')

    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=True)

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
        '--seg_model',
        default='deeplab',
        type=str,
        help='(deeplab)')

    parser.add_argument(
        '--cla_model',
        default='resnet',
        type=str,
        help='(deeplab)')

    parser.add_argument(
        '--cla_model_depth',
        default=50,
        type=int,
        help='34|50|101|')

    parser.add_argument(
        '--manual_seed', default=1, type=int, 
        help='Manually set random seed')

    args = parser.parse_args()

    return args
