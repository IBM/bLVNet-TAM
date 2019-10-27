import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action Recognition Training')

    # model definition
    parser.add_argument('-d', '--depth', default=50, type=int, metavar='N',
                        help='depth of blresnet (default: 50)', choices=[50, 101])
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--groups', default=16, type=int)
    parser.add_argument('--frames_per_group', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=int, metavar='N', help='ratio of channels')
    parser.add_argument('--beta', default=4, type=int, metavar='N', help='ratio of layers')
    parser.add_argument('--blending_frames', default=3, type=int)
    # training setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str,
                        help='learning rate scheduler', choices=['step', 'multisteps', 'cosine', 'plateau'])
    parser.add_argument('--lr_steps', default=[15, 30, 45], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--imagenet_blnet_pretrained', action='store_true',
                        help='use imagenet-pretrained blnet model')

    # data-related
    parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--dataset', default='st2stv2',
                        choices=['st2stv2', 'st2stv1', 'kinetics400', 'moments_30fps'],
                        help='path to dataset file list')
    parser.add_argument('--input_shape', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, directly crop the input_shape size')
    parser.add_argument('--random_sampling', action='store_true', help='perform determinstic sampling for data loader')
    parser.add_argument('--dense_sampling', action='store_true', help='perform dense sampling for data loader')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow', choices=['rgb', 'flow'])
    # logging
    parser.add_argument('--logdir', default='', type=str, help='log path')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='frequency to print the log during the training')
    parser.add_argument('--show_model', action='store_true', help='show model summary')

    # for testing
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    parser.add_argument('--num_clips', default=1, type=int)
    return parser
