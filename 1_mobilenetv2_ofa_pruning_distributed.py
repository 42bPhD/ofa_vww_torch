# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.mobilenet import mobilenet_v2

from pytorch_nndct import OFAPruner
from utils.utils import AverageMeter, ProgressMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch OFA ImageNet Training')

parser.add_argument(
    '--data',
    type=str,
    default="E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96",
    help='path to dataset')

# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=8, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.25, 0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes',
    type=list,
    default=None,
    help='excludes module')

parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=25,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256), this is the total '
    'batch size of all GPUs on the current node when '
    'using Data Parallel or Distributed Data Parallel')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
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
    default=100,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--seed', default=20, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=True, type=int, help='GPU id to use.')

best_acc1 = 0

def sgd_optimizer(model, lr, momentum, weight_decay):
  params = []
  for key, value in model.named_parameters():
    if not value.requires_grad:
      continue
    apply_weight_decay = weight_decay
    apply_lr = lr
    if 'bias' in key or 'bn' in key:
      apply_weight_decay = 0
    if 'bias' in key:
      apply_lr = 2 * lr
    params += [{
        'params': [value],
        'lr': apply_lr,
        'weight_decay': apply_weight_decay
    }]
  optimizer = torch.optim.SGD(params, lr, momentum=momentum)
  return optimizer

def main():
  args = parser.parse_args()

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                  'disable data parallelism.')

  ngpus_per_node = torch.cuda.device_count()
  
  # Simply call main_worker function
  main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
  global best_acc1
  args.gpu = gpu

  if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

  if not torch.cuda.is_available():
    print('using CPU, this will be slow')
  elif args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)

      args.teacher_model = mobilenet_v2(pretrained=True)

      args.teacher_model.cuda(args.gpu)

      inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32).cuda()

      ofa_pruner = OFAPruner(args.teacher_model, inputs)
      ofa_model = ofa_pruner.ofa_model(args.expand_ratio,
                                       args.channel_divisible, args.excludes)

      model = ofa_model.cuda(args.gpu)

      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs we have
      args.batch_size = int(args.batch_size)
      args.workers = int(args.workers)
      model = model.cuda()
    else:
      raise ValueError('gpu is None')
      
  # define loss function (criterion and kd loss) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)
  soft_criterion = cross_entropy_loss_with_soft_target().cuda(args.gpu)

  optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay)
  VWW_TRAINSET_SIZE = 10000
  lr_scheduler = CosineAnnealingLR(
      optimizer=optimizer,
      T_max=args.epochs * VWW_TRAINSET_SIZE // args.batch_size //
      ngpus_per_node)

  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      
      # Map model to be loaded to specified single gpu.
      loc = 'cuda:{}'.format(args.gpu)
      checkpoint = torch.load(args.resume, map_location=loc)
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      if args.gpu is not None:
        # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(args.gpu)
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})".format(
          args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  traindir = os.path.join(args.data, 'train')
  valdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
          transforms.RandomResizedCrop(96),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))

  train_sampler = None

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=(train_sampler is None),
      num_workers=args.workers,
      pin_memory=True,
      sampler=train_sampler)

  val_dataset = datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.Resize(96),
          transforms.CenterCrop(96),
          transforms.ToTensor(),
          normalize,
      ]))
  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.workers,
      pin_memory=True)

  for epoch in range(args.start_epoch, args.epochs):
    
    if epoch == args.start_epoch:
      validate_subnet(train_loader, val_loader, model, ofa_pruner, criterion,
                      args, False)

    # train for one epoch
    train(train_loader, model, ofa_pruner, criterion, soft_criterion, optimizer,
          epoch, args, lr_scheduler)

    # evaluate on validation set
    validate_subnet(train_loader, val_loader, model, ofa_pruner, criterion,
                    args, True)

    
    save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_acc1': best_acc1,
          'optimizer': optimizer.state_dict(),
          'scheduler': lr_scheduler.state_dict(),
      })

def train(train_loader, model, ofa_pruner, criterion, soft_criterion, optimizer,
          epoch, args, lr_scheduler):

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

    if args.gpu is not None:
      images = images.cuda(args.gpu, non_blocking=True)
    if torch.cuda.is_available():
      target = target.cuda(args.gpu, non_blocking=True)

    # total subnets to be sampled
    optimizer.zero_grad()

    args.teacher_model.train()
    with torch.no_grad():
      #! TODO: check if this is the correct way to detach the logits
      soft_logits = args.teacher_model(images).detach()

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

    optimizer.step()
    lr_scheduler.step()

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
    if i % 1000 == 0:
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
          images = images.cuda(non_blocking=True)
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
      if args.gpu is not None:
        images = images.cuda(args.gpu, non_blocking=True)
      if torch.cuda.is_available():
        target = target.cuda(args.gpu, non_blocking=True)

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

class cross_entropy_loss_with_soft_target(torch.nn.modules.loss._Loss):
  def forward(self, output, target):
    target = torch.nn.functional.softmax(target, dim=1)
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-target * logsoftmax(output), 1))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
  torch.save(state, filename)
  shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
  main()
