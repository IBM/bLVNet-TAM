import os
import time

from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

from core.video_utils import build_dataflow, build_model
from core.video_transforms import *
from core.video_dataset import VideoDataSet
from opts import arg_parser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        cls_id = 0
        for label in f.readlines():
            label = label.strip()
            if label == "":
                continue
            id_to_label[cls_id] = label
            label_to_id[label] = cls_id
            cls_id += 1
    return id_to_label, label_to_id


def eval_a_batch(data, model, num_clips=1, num_crops=1, softmax=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result = model(data)
        result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
        if softmax:
            # take the softmax to normalize the output to probability
            result = F.softmax(result, dim=1)

        return result


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True
    id_to_label = {}

    if args.dataset == 'st2stv2':
        num_classes = 174
        data_list_name = 'validation_256.txt' if args.evaluate else 'testing_256.txt'
        filename_seperator = " "
        image_tmpl = '{:05d}.jpg'
        filter_video = 3
    elif args.dataset == 'st2stv1':
        num_classes = 174
        data_list_name = 'validation_256.txt' if args.evaluate else 'testing_256.txt'
        filename_seperator = " "
        image_tmpl = '{:05d}.jpg'
        label_file = 'something-something-v1-labels.csv'
        filter_video = 3
        id_to_label, label_to_id = load_categories(os.path.join(args.datadir, label_file))
    else:  # 'kinetics400'
        num_classes = 400
        data_list_name = 'val_400_331.txt' if args.evaluate else 'test_400_331.txt'
        filename_seperator = ";"
        image_tmpl = '{:05d}.jpg'
        filter_video = 30

    args.num_classes = num_classes

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:  # flow
        mean = [0.5]
        std = [np.mean([0.229, 0.224, 0.225])]

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args, test_mode=True)
    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
    else:
        print("=> creating model '{}'".format(arch_name))

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_shape
    else:
        scale_size = int(args.input_shape / 0.875 + 0.5)

    augments = []
    if args.num_crops == 1:
        augments += [
            GroupScale(scale_size),
            GroupCenterCrop(args.input_shape)
        ]
    else:
        flip = True if args.num_crops == 10 else False
        augments += [
            GroupOverSample(args.input_shape, scale_size, num_crops=args.num_crops, flip=flip),
        ]
    augments += [
        Stack(),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        GroupNormalize(mean=mean, std=std)
    ]

    augmentor = transforms.Compose(augments)

    # Data loading code
    data_list = os.path.join(args.datadir, data_list_name)
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_shape))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}, offset from center with {}".format(args.num_clips, sample_offsets))

    val_dataset = VideoDataSet("", data_list, args.groups, args.frames_per_group,
                               num_clips=args.num_clips, modality=args.modality,
                               image_tmpl=image_tmpl,
                               dense_sampling=args.dense_sampling,
                               fixed_offset=not args.random_sampling,
                               transform=augmentor, is_train=False, test_mode=not args.evaluate,
                               seperator=filename_seperator, filter_video=filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    batch_time = AverageMeter()
    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
        top1 = AverageMeter()
        top5 = AverageMeter()
    else:
        logfile = open(os.path.join(log_folder,
                                    'test_{}crops_{}clips_{}.csv'.format(args.num_crops,
                                                                         args.num_clips,
                                                                         args.input_shape)), 'w')

    total_outputs = 0
    outputs = np.zeros((len(data_loader) * args.batch_size, num_classes))
    # switch to evaluate mode
    model.eval()
    total_batches = len(data_loader)
    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        for i, (video, label) in enumerate(data_loader):
            output = eval_a_batch(video, model, num_clips=args.num_clips, num_crops=args.num_crops,
                                  softmax=True)
            if args.evaluate:
                label = label.cuda(non_blocking=True)
                # measure accuracy
                prec1, prec5 = accuracy(output, label, topk=(1, 5))
                top1.update(prec1[0], video.size(0))
                top5.update(prec5[0], video.size(0))
                output = output.data.cpu().numpy().copy()
                batch_size = output.shape[0]
                outputs[total_outputs:total_outputs + batch_size, :] = output
            else:
                # testing, store output to prepare csv file
                # measure elapsed time
                output = output.data.cpu().numpy().copy()
                batch_size = output.shape[0]
                outputs[total_outputs:total_outputs + batch_size, :] = output
                predictions = np.argsort(output, axis=1)
                for ii in range(len(predictions)):
                    temp = predictions[ii][::-1][:5]
                    preds = [str(pred) for pred in temp]
                    if args.dataset == 'st2stv1':
                        print("{};{}".format(label[ii], id_to_label[int(preds[0])]), file=logfile)
                    else:
                        print("{};{}".format(label[ii], ";".join(preds)), file=logfile)
            total_outputs += video.shape[0]
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

        # if not args.evaluate:
        outputs = outputs[:total_outputs]
        print("Predict {} videos.".format(total_outputs), flush=True)
        np.save(os.path.join(log_folder, '{}_{}crops_{}clips_{}_details.npy'.format(
            "val" if args.evaluate else "test", args.num_crops, args.num_clips, args.input_shape)),
                outputs)

    if args.evaluate:
        print(
            'Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\t'.format(
                args.input_shape, scale_size, args.num_crops, args.num_clips, top1.avg, top5.avg
                ), flush=True)
        print(
            'Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\t'.format(
                args.input_shape, scale_size, args.num_crops, args.num_clips, top1.avg, top5.avg
                ), flush=True, file=logfile)

    logfile.close()


if __name__ == '__main__':
    main()
