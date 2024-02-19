#!/usr/bin/env bash

import argparse
import os
from pytorch_nndct import get_pruning_runner
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.trains_evals import train, evaluate
from utils.utils import eval_fn, get_gpus
from utils.io import load_weights
from utils.dataloaders import get_dataloader
from models.vww_model import mobilenet_v1
from utils.io import StepMonitor

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', 
                    type=str, 
                    default='0',
                    help='String of available GPU number')
parser.add_argument('--lr', 
                    type=float, 
                    default=1e-3, 
                    help='Initial learning rate')
parser.add_argument('--epochs', 
                    type=int, 
                    default=1, 
                    help='Train epoch')
parser.add_argument('--subnet_index',
                    default=None,
                    help='Choose which subnet to retrain')
parser.add_argument('--sparsity',
                    type=float,
                    default=0.5,
                    help='Sparsity ratio')
parser.add_argument('--pretrained',
                    type=str,
                    default='E:\\1_TinyML\\embedded\\trained_models\\vww_baseline_912am2ws\\vww_baseline_best.pth',
                    help='Pretrained model filepath')
parser.add_argument('--save_dir',
                    type=str,
                    default='./trained_models/one_step_pruned',
                    help='Where to save retrained model')
parser.add_argument('--data_dir',
                    type=str,
                    default='E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96',
                    help='Dataset directory')
parser.add_argument('--num_workers',
                    type=int,
                    default=4,
                    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=50, 
                    help='Batch size')
parser.add_argument("--image-size", 
                    default=96, 
                    type=int, 
                    help="Input image size (square assumed).")
parser.add_argument('--weight_decay', 
                    type=float, 
                    default=1e-4, 
                    help='Weight decay')
parser.add_argument('--momentum', 
                    type=float, 
                    default=0.9, 
                    help='Momentum')
args, _ = parser.parse_known_args()

device = f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu'
gpus = get_gpus(args.gpus)

if __name__ == '__main__':
    assert os.path.exists(args.pretrained), "No pretrained model!"
    if not os.path.exists(args.data_dir):
        assert False, "No dataset found!"

    train_loader ,val_loader = get_dataloader(args.data_dir, 
                                              args.batch_size, 
                                              num_workers=args.num_workers, 
                                              shuffle=True, 
                                              image_size=args.image_size)

    model = mobilenet_v1()
    model = load_weights(model, args.pretrained)
    input_signature = torch.randn([1, 3, args.image_size, args.image_size],
                                  dtype=torch.float32)
    input_signature = input_signature.to(device)
    model = model.to(device)
    pruning_runner = get_pruning_runner(model, 
                                        input_signature, 
                                        'one_step')
    # index=None: select optimal subnet automatically
    model = pruning_runner.prune(removal_ratio=args.sparsity, mode='slim', index=args.subnet_index)
    model = model.to(device)
    step_Monitor = StepMonitor(model_name = args.save_dir)
    save_path = 'model_slim.pth'

    # exit()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 args.lr, 
                                 weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_acc1 = 0
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        acc1, acc5 = evaluate(val_loader, model, criterion)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            torch.onnx.export(model, 
                              torch.randn(1, 3, args.image_size, args.image_size).to(device), 
                              os.path.join(step_Monitor.version, 'model.onnx'))
            step_Monitor.save_checkpoint(state = {
                    'epoch': epoch + 1,
                    'state_dict': model.cpu().state_dict(),
                    'best_acc1': best_acc1,
                    'sparse': 'one_step',
                    'sparse_ratio': args.sparsity,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    }, 
                    filename=save_path)