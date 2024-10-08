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
import time
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_nndct import OFAPruner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=2, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes', type=list, default=None, help='excludes module')

parser.add_argument(
    '--targeted_macs',
    type=int,
    default=290,
    help='targeted macs subnet need to be export')

parser.add_argument(
    '--pretrained_ofa_model',
    type=str,
    default='mynet_pruned.pth',
    help='Pretrained ofa model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./dataset/cifar10',
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=4,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

args, _ = parser.parse_known_args()

from utils.utils import AverageMeter, ProgressMeter, accuracy, get_gpus

def evaluate(dataloader, model, criterion):
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

  return float(top1.avg), float(top5.avg)

class MyNet(nn.Module):

  def __init__(self):
    super(MyNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(32, 128, 3, stride=2)
    self.bn2 = nn.BatchNorm2d(128)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv3 = nn.Conv2d(128, 256, 3)
    self.bn3 = nn.BatchNorm2d(256)
    self.relu3 = nn.ReLU(inplace=True)
    self.conv4 = nn.Conv2d(256, 512, 3)
    self.bn4 = nn.BatchNorm2d(512)
    self.relu4 = nn.ReLU(inplace=True)
    self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.Linear(512, 32)
    self.fc = nn.Linear(32, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.relu4(x)
    x = self.avgpool1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc(x)
    return x

if __name__ == '__main__':

  gpus = get_gpus(args.gpus)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MyNet()

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  train_dataset = datasets.CIFAR10(
      root=args.data_dir, train=True, download=True, transform=transform)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers)

  val_dataset = datasets.CIFAR10(
      root=args.data_dir, train=False, download=True, transform=transform)
  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers)

  model.to(device)

  input_signature = torch.randn([1, 3, 32, 32], dtype=torch.float32).cuda()

  ofa_pruner = OFAPruner(model, input_signature)
  ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                   args.excludes)

  model = ofa_model.cuda()

  #reloading model
  with open(args.pretrained_ofa_model, 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')
  assert isinstance(checkpoint, dict)
  for k, v in model.state_dict().items():
    v.copy_(checkpoint[k])

  pareto_global = ofa_pruner.load_subnet_config('cifar10_pareto_global.txt')
  
  target_macs = min(map(int, list(pareto_global.keys())))
  
  static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
      ofa_model, pareto_global[str(target_macs)]['subnet_setting'])

  model = static_subnet

  model.cuda()

  with torch.no_grad():
    model.eval()
    ofa_pruner.reset_bn_running_stats_for_calibration(model)
    for batch_idx, (images, _) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      model(images)
      if batch_idx >= 16:
        break

  criterion = torch.nn.CrossEntropyLoss().cuda()

  top1, top5 = evaluate(val_loader, model, criterion)

  print('subnet Acc@top1:', top1)
  print('subnet macs:', macs)
  print('subnet params:', params)
