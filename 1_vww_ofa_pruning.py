
import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision.models.mobilenet import mobilenet_v2

from vww_model import mobilenet_v1 as mobilenet_v2
# from pytorch_nndct import OFAPruner

parser = argparse.ArgumentParser(description='PyTorch OFA VWW Training')

# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=2, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes', type=list, default=None, help='excludes module')
parser.add_argument(
    '--pretrained',
    type=str,
    default='vww_trained.pth',
    help='Pretrained model filepath')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    help="Number of epochs. Default value is 500 according to TF training procedure.",
)
parser.add_argument(
    "--data-dir",
    default="E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96",
    type=str,
    help="Path to dataset (will be downloaded).",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=50,
    help="Batch size. Default value is 32 according to TF training procedure.",
)
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=1e-3,
    type=float,
    metavar='LR',
    help='initial learning rate',
    dest='lr')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--wd',
    '--weight-decay',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)',
    dest='weight_decay')
parser.add_argument(
    '-p',
    '--print-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
        "--image-size", default=96, type=int, help="Input image size (square assumed)."
    )
parser.add_argument(
    "--save-path", default="trained_models", type=str, help="Path to save model."
)
parser.add_argument(
    '--seed', default=20, type=int, help='seed for initializing training. ')


args, _ = parser.parse_known_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by every 2 epochs"""
    lr = lr * (0.1**(epoch // 2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          teacher_model=None,
          ofa_pruner=None,
          soft_criterion=None,
          args=None,
          lr_scheduler=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), [
            batch_time,
            data_time,
            losses,
            top1,
            top5,
        ],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        model = model.cuda()
        images = images.cuda()
        target = target.cuda()

        # total subnets to be sampled
        optimizer.zero_grad()
        if teacher_model:
            teacher_model.train()
            with torch.no_grad():
                soft_logits = teacher_model(images).detach()

        if ofa_pruner:
            for arch_id in range(4):
                if arch_id == 0:
                    model, _ = ofa_pruner.sample_subnet(model, 'max')
                elif arch_id == 1:
                    model, _ = ofa_pruner.sample_subnet(model, 'min')
                else:
                    model, _ = ofa_pruner.sample_subnet(model, 'random')

                # calcualting loss
                output = model(images)

                if soft_criterion:
                    loss = soft_criterion(output, soft_logits) + criterion(output, target)
                else:
                    loss = criterion(output, target)

                loss.backward()
        else:
            # compute output
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.display(i)
        
        if lr_scheduler and i % 1000==0:
            print('cur lr: ', lr_scheduler.get_lr()[0])

def validate_subnet(train_loader, val_loader, model, ofa_pruner, criterion,
                    args, bn_calibration):

    evaluated_subnet = {
        'ofa_min_subnet': {},
        'ofa_max_subnet': {},
    }

    for net_id in evaluated_subnet:
        if net_id == 'ofa_min_subnet':
            dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
                model, 'min')
        elif net_id == 'ofa_max_subnet':
            dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
                model, 'max')
        else:
            dynamic_subnet_setting = evaluated_subnet[net_id]
            static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
                    model, dynamic_subnet_setting)

        if len(evaluated_subnet[net_id]) == 0:
            static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
                dynamic_subnet, dynamic_subnet_setting)

        static_subnet = static_subnet.cuda()

        if bn_calibration:
            with torch.no_grad():
                static_subnet.eval()
                ofa_pruner.reset_bn_running_stats_for_calibration(static_subnet)

                for batch_idx, (images, _) in enumerate(train_loader):
                    if batch_idx >= 16:
                        break
                images = images.cuda()
                static_subnet(images)  #forward only

    acc1, acc5 = validate(val_loader, static_subnet, criterion, args)

    summary = {
        'net_id': net_id,
        'mode': 'evaluate',
        'acc1': acc1,
        'acc5': acc5,
        'macs': macs,
        'params': params
    }

    print(summary)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(args.gpu)
                target = target.cuda(args.gpu)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

    return float(top1.avg), float(top5.avg)

def evaluate(dataloader, model, criterion, ofa_pruner=None, train_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    if ofa_pruner:
        dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
            model, 'max')
        static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
            dynamic_subnet, dynamic_subnet_setting)

        model = static_subnet.cuda()

        with torch.no_grad():
            model.eval()
            ofa_pruner.reset_bn_running_stats_for_calibration(model)

            print('runing bn calibration...')

            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.cuda(non_blocking=True)
                model(images)
                if batch_idx >= 16:
                    break

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(dataloader):
            model = model.cuda()
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

    return top1.avg, top5.avg

class cross_entropy_loss_with_soft_target(torch.nn.modules.loss._Loss):
  def forward(self, output, target):
    target = torch.nn.functional.softmax(target, dim=1)
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-target * logsoftmax(output), 1))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    shutil.copyfile(filename, 'model_best.pth.tar')

def load_weights(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model

from utils.dataloaders import get_dataloader
if __name__ == '__main__':
    model = mobilenet_v2()
    model.cuda()
    train_loader, val_loader = get_dataloader(dataset_dir=args.data_dir, 
                                            batch_size=args.batch_size, 
                                            image_size=args.image_size, 
                                            num_workers=args.workers, 
                                            shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    if os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained)
        model.load_state_dict(ckpt)
        acc1, acc5 = validate(val_loader, model, criterion, args)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay)

    best_acc1 = 0
    
    for epoch in range(args.epoches):
        adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, optimizer, epoch)
        acc1, acc5 = validate(val_loader, model, criterion)
        if acc1 > best_acc1:
            best_acc1 = acc1
            with torch.no_grad():
                torch.save(model.cpu().state_dict(), args.pretrained)
                model = model.cuda()
    
    
    teacher_model = model
    inputs = torch.randn([1, 3, 96, 96], dtype=torch.float32).cuda()

    
    # ofa_pruner = OFAPruner(teacher_model, inputs)
    # ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
    #                                 args.excludes)
    
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay)

    soft_criterion = cross_entropy_loss_with_soft_target().cuda(args.gpu)