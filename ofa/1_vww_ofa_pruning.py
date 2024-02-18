
import argparse
import os
import time
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision.models.mobilenet import mobilenet_v2
from utils.utils import (save_checkpoint, 
                            load_weights, 
                            adjust_learning_rate, 
                            accuracy)
from utils.utils import Cross_entropy_loss_with_soft_target
from utils.common import AverageMeter, ProgressMeter
from utils.dataloaders import get_dataloader

from models.vww_model import mobilenet_v1 as mobilenet_v2
from pytorch_nndct import OFAPruner

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
                teacher_model.cuda()
                soft_logits = teacher_model(images)

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
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
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

def validate_subnet(train_loader, 
                    val_loader, 
                    model, 
                    ofa_pruner, 
                    criterion,
                    args:dict, 
                    bn_calibration:bool):

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
            print(f'{net_id} model runing bn calibration...')
            with torch.no_grad():
                static_subnet.eval()
                ofa_pruner.reset_bn_running_stats_for_calibration(static_subnet)

                for batch_idx, (images, _) in enumerate(train_loader):
                    if batch_idx >= 16:
                        break
                images = images.cuda()
                static_subnet(images)  #forward only

    acc1, acc5 = validate_subnet(val_loader, static_subnet, criterion, args)

    summary = {
        'net_id': net_id,
        'mode': 'evaluate',
        'acc1': acc1,
        'acc5': acc5,
        'macs': macs,
        'params': params
    }

    print(summary)

def evaluate(val_loader, model, criterion, ofa_pruner=None, train_loader=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    if ofa_pruner:
        dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
            model.cpu(), 'max')
        static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
            dynamic_subnet, dynamic_subnet_setting)

        model = static_subnet.to(device)

        with torch.no_grad():
            model.eval()
            ofa_pruner.reset_bn_running_stats_for_calibration(model)

            print('runing bn calibration...')

            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(device)
                model(images)
                if batch_idx >= 16:
                    break

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            model = model.to(device)
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
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



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2(pretrained=True)
    #The last linear layer changed to 2
    # model.classifier[1] = nn.Linear(1280, 2)
    
    
    model.cuda()
    train_loader, val_loader = get_dataloader(dataset_dir=args.data_dir, 
                                            batch_size=args.batch_size, 
                                            image_size=args.image_size, 
                                            num_workers=args.workers, 
                                            shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler = CosineAnnealingLR(
      optimizer=optimizer,
      T_max=int(args.epochs) * len(train_loader) // args.batch_size)
    
    soft_criterion = Cross_entropy_loss_with_soft_target().cuda(device)
        
    if os.path.exists(args.pretrained):
        model = load_weights(model, args.pretrained)
        acc1, acc5 = evaluate(val_loader=val_loader, 
                              model=model, 
                              criterion=criterion, 
                              ofa_pruner=None, train_loader=None)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay)
        best_acc1 = 0
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr)
            train(train_loader, model, criterion, optimizer, epoch)  
            acc1, acc5 =  evaluate(val_loader=val_loader, 
                                model=model, 
                                criterion=criterion, 
                                ofa_pruner=None, train_loader=None)
            if acc1 > best_acc1:
                best_acc1 = acc1
                with torch.no_grad():
                    save_checkpoint(state = {
                    'epoch': epoch + 1,
                    'state_dict': model.cpu().state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    }, 
                    filename=args.pretrained)
                    model = model.cuda()
                    
    
    teacher_model = model
    inputs = torch.randn([1, 3, 96, 96], dtype=torch.float32)

    
    ofa_pruner = OFAPruner(teacher_model, inputs)
    ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                    args.excludes)
    model = ofa_model.cuda()
    
    best_acc1 = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, 
                model, 
                criterion, 
                optimizer, 
                epoch, 
                teacher_model,
                ofa_pruner, soft_criterion)
        acc1, acc5 =  evaluate(val_loader=val_loader, 
                              model=model, 
                              criterion=criterion, 
                              ofa_pruner=ofa_pruner, train_loader=train_loader)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            pretrained_path = ''.join(os.path.splitext(args.pretrained)[0] + '_best' + os.path.splitext(args.pretrained)[1])
            save_checkpoint(state = {
                'epoch': epoch + 1,
                'state_dict': model.cpu().state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                }, 
                filename=pretrained_path)
            model = model.cuda()