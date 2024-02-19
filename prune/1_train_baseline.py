#!/usr/bin/env bash

import argparse
import os

import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from data import get_dataloader
from utils.dataloaders import get_dataloader
# from net import MyNet
from models.vww_model import mobilenet_v1
from torch.optim.lr_scheduler import CosineAnnealingLR


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
                    default=50, 
                    help='Train epoch')
parser.add_argument('--pretrained',
                    type=str,
                    default='vww_baseline.pth',
                    help='Pretrained model filepath')
parser.add_argument('--data_dir',
                    type=str,
                    default='E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96',
                    help='Dataset directory')
parser.add_argument('--save_dir',
                    type=str,
                    default='trained_models',
                    help='Dataset directory')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=50, 
                    help='Batch size')
parser.add_argument('--weight_decay', 
                    type=float, 
                    default=1e-4, 
                    help='Weight decay')
parser.add_argument('--momentum', 
                    type=float, 
                    default=0.9, 
                    help='Momentum')

args, _ = parser.parse_known_args()


from utils.trains_evals import train, evaluate
from utils.utils import adjust_learning_rate, save_checkpoint
from utils.io import StepMonitor
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = mobilenet_v1().to(device)
    
    batch_size = args.batch_size

    train_loader, val_loader = get_dataloader(args.data_dir, 
                                              batch_size, 
                                              num_workers=args.num_workers, 
                                              shuffle=True)
    
    step_Monitor = StepMonitor(model_name = './trained_models/vww_baseline')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if not os.path.exists(args.pretrained):
        optimizer = torch.optim.Adam(model.parameters(), 
                                     args.lr, 
                                     weight_decay=args.weight_decay)
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs)
        best_acc1 = 0
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr)
            train(train_loader, model, criterion, optimizer, epoch)
            lr_scheduler.step()
            acc1, acc5 = evaluate(val_loader, model, criterion)
            if acc1 > best_acc1:
                best_acc1 = acc1
                pretrained_path = ''.join(os.path.splitext(args.pretrained)[0] + '_best' + os.path.splitext(args.pretrained)[1])
                
                step_Monitor.save_checkpoint(state = {
                    'epoch': epoch + 1,
                    'state_dict': model.cpu().state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    }, 
                    filename=pretrained_path)   
