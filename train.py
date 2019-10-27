import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
import tensorboard_logger

from core.video_utils import (train, validate, build_dataflow, get_augmentor, build_model)
from core.video_dataset import VideoDataSet
from opts import arg_parser


def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True

    if args.dataset == 'st2stv2':
        num_classes = 174
        train_list_name = 'training_256.txt'
        val_list_name = 'validation_256.txt'
        filename_seperator = " "
        image_tmpl = '{:05d}.jpg'
        filter_video = 3
    elif args.dataset == 'st2stv1':
        num_classes = 174
        train_list_name = 'training_256.txt'
        val_list_name = 'validation_256.txt'
        filename_seperator = " "
        image_tmpl = '{:05d}.jpg'
        filter_video = 3
    else:  # kinetics400
        num_classes = 400
        train_list_name = 'train_400_331.txt'
        val_list_name = 'val_400_331.txt'
        filename_seperator = ";"
        image_tmpl = '{:05d}.jpg'
        filter_video = 30
    # elif args.dataset == 'moments_30fps':
    #     num_classes = 339
    #     train_list_name = 'training_256.txt'
    #     val_list_name = 'validation_256.txt'
    #     filename_seperator = " "
    #     image_tmpl = '{:05d}.jpg'

    args.num_classes = num_classes

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.modality == 'flow':
        mean = [0.5]
        std = [np.mean([0.229, 0.224, 0.225])]

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args)

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
    else:
        print("=> creating model '{}'".format(arch_name))

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss().cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()

    # Data loading code
    val_list = os.path.join(args.datadir, val_list_name)

    val_augmentor = get_augmentor(False, args.input_shape, mean=mean, std=std,
                                  disable_scaleup=args.disable_scaleup,
                                  is_flow=True if args.modality == 'flow' else False)

    val_dataset = VideoDataSet("", val_list, args.groups, args.frames_per_group,
                               num_clips=args.num_clips,
                               modality=args.modality, image_tmpl=image_tmpl,
                               dense_sampling=args.dense_sampling,
                               transform=val_augmentor, is_train=False, test_mode=False,
                               seperator=filename_seperator, filter_video=filter_video,
                               num_classes=args.num_classes)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion)
        print(
            'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                args.input_shape, val_losses, val_top1, val_top5, val_speed * 1000.0), flush=True)
        print(
            'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                args.input_shape, val_losses, val_top1, val_top5, val_speed * 1000.0), flush=True, file=logfile)
        return

    train_list = os.path.join(args.datadir, train_list_name)

    train_augmentor = get_augmentor(True, args.input_shape, mean=mean, std=std,
                                    disable_scaleup=args.disable_scaleup,
                                    is_flow=True if args.modality == 'flow' else False)

    train_dataset = VideoDataSet("", train_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=train_augmentor, is_train=True, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video,
                                 num_classes=args.num_classes)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_steps[0], gamma=0.1)
    elif args.lr_scheduler == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    elif args.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    best_top1 = 0.0
    tensorboard_logger.configure(os.path.join(log_folder))
    # optionally resume from a checkpoint
    if args.resume:
        logfile = open(os.path.join(log_folder, 'log.log'), 'a')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.lr_scheduler == 'plateau':
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if os.path.exists(os.path.join(log_folder, 'log.log')):
            shutil.copyfile(os.path.join(log_folder, 'log.log'), os.path.join(
                log_folder, 'log.log.{}'.format(int(time.time()))))
        logfile = open(os.path.join(log_folder, 'log.log'), 'w')

    print(args, flush=True)
    print(model, flush=True)

    print(args, file=logfile, flush=True)

    if args.resume is None:
        print(model, file=logfile, flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_losses, epoch)
        else:
            scheduler.step(epoch)
        try:
            # get_lr get all lrs for every layer of current epoch, assume the lr for all layers are identical
            lr = scheduler.optimizer.param_groups[0]['lr']
        except:
            lr = None
        # set current learning rate
        # train for one epoch
        train_top1, train_top5, train_losses, train_speed, speed_data_loader, train_steps = \
            train(train_loader, model, train_criterion, optimizer, epoch + 1, display=args.print_freq)
        print(
            'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                speed_data_loader * 1000.0), file=logfile, flush=True)
        print(
            'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                speed_data_loader * 1000.0), flush=True)

        # evaluate on validation set
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion)
        print(
            'Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0),
            file=logfile, flush=True)
        print(
            'Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0),
            flush=True)
        # remember best prec@1 and save checkpoint
        is_best = val_top1 > best_top1
        best_top1 = max(val_top1, best_top1)

        save_dict = {'epoch': epoch + 1,
                     'arch': arch_name,
                     'state_dict': model.state_dict(),
                     'best_top1': best_top1,
                     'optimizer': optimizer.state_dict(),
                     }
        if args.lr_scheduler == 'plateau':
            save_dict['scheduler'] = scheduler.state_dict()

        save_checkpoint(save_dict, is_best, filepath=log_folder)

        if lr is not None:
            tensorboard_logger.log_value('learning-rate', lr, epoch + 1)
        tensorboard_logger.log_value('val-top1', val_top1, epoch + 1)
        tensorboard_logger.log_value('val-loss', val_losses, epoch + 1)
        tensorboard_logger.log_value('train-top1', train_top1, epoch + 1)
        tensorboard_logger.log_value('train-loss', train_losses, epoch + 1)
        tensorboard_logger.log_value('best-val-top1', best_top1, epoch + 1)

    logfile.close()


if __name__ == '__main__':
    main()
