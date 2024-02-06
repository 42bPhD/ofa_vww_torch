import argparse
import os

import torch

from data import get_dataloader_ddp, get_dataloader, get_subnet_dataloader
from net import MyNet
from utils import *

from pytorch_nndct import get_pruning_runner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--local_rank', type=int, default=0, help='node rank for distributed training')
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
parser.add_argument(
    '--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=1, help='Train epoch')
parser.add_argument(
    '--subnet_index',
    default=None,
    help='Choose which subnet to retrain'
)
parser.add_argument(
    '--sparsity', type=float, default=0.5, help='Sparsity ratio')
parser.add_argument(
    '--pretrained',
    type=str,
    default='mynet.pth',
    help='Pretrained model filepath')
parser.add_argument(
    '--save_dir',
    type=str,
    default='./',
    help='Where to save retrained model')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./dataset/cifar10',
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=48,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument(
    '--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
args, _ = parser.parse_known_args()

device = 'cuda'
gpus = get_gpus(args.gpus)

if __name__ == '__main__':
  assert os.path.exists(args.pretrained), "No pretrained model!"
  if os.path.exists(args.data_dir):
    download = False
  else:
    download = True

  torch.distributed.init_process_group(backend="nccl")
  torch.cuda.set_device(args.local_rank)
  batch_size = args.batch_size * len(gpus)
  train_loader = get_dataloader_ddp(args.data_dir, batch_size, num_workers=args.num_workers, shuffle=True, train=True, download=download)
  val_loader = get_dataloader_ddp(args.data_dir, batch_size, num_workers=args.num_workers, shuffle=False, train=False, download=download)

  model = MyNet()
  model = load_weights(model, args.pretrained)
  input_signature = torch.randn([1, 3, 32, 32], dtype=torch.float32)
  input_signature = input_signature.to(device)
  model = model.to(device)
  pruning_runner = get_pruning_runner(model, input_signature, 'one_step')
  # index=None: select optimal subnet automatically
  model = pruning_runner.prune(removal_ratio=args.sparsity, mode='slim', index=args.subnet_index)

  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.Adam(
      model.parameters(), args.lr, weight_decay=args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
  best_acc1 = 0
  for epoch in range(args.epochs):
    train(train_loader, model, criterion, optimizer, epoch)
    lr_scheduler.step()
    acc1, acc5 = evaluate(val_loader, model, criterion)
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if is_best:
      if hasattr(model, 'module'):
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'mynet_slim.pth'))
      else:
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'mynet_slim.pth'))
