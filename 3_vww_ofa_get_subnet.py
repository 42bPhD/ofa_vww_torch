
import argparse
import time
import torch
from pytorch_nndct import OFAPruner

import argparse
import time
import torch
# from torchvision.models.mobilenet import mobilenet_v2
from pytorch_nndct import OFAPruner
from utils.utils import get_gpus, AverageMeter, ProgressMeter, accuracy

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=2, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.25, 0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes',
    type=list,
    default=None,
    help='excludes module')

parser.add_argument(
    '--targeted_macs',
    type=int,
    default=240,
    help='targeted macs subnet need to be export')

parser.add_argument(
    '--pretrained_ofa_model',
    type=str,
    default='vww_trained_best.pth',
    help='Pretrained ofa model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default='E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96',
    help='Dataset directory')
parser.add_argument(
    '--workers',
    type=int,
    default=4,
    help='Number of workers used in dataloading')
parser.add_argument(
        "--image-size", default=96, type=int, help="Input image size (square assumed)."
    )
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')

args, _ = parser.parse_known_args()


def evaluate(dataloader, model, criterion):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(dataloader), [batch_time, losses, top1, top5], prefix='Test: ')

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

from utils.utils import get_gpus, load_weights
from utils.dataloaders import get_dataloader
from vww_model import mobilenet_v1 as mobilenet_v2
if __name__ == '__main__':
    gpus = get_gpus(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2()
    train_loader, val_loader = get_dataloader(dataset_dir=args.data_dir, 
                                            batch_size=args.batch_size, 
                                            image_size=args.image_size, 
                                            num_workers=args.workers, 
                                            shuffle=True)

    model.to(device)

    input_signature = torch.randn([1, 3, args.image_size, args.image_size], dtype=torch.float32).cuda()

    ofa_pruner = OFAPruner(model, input_signature)
    ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                    args.excludes)

    model = ofa_model.cuda()

    #reloading model
    # with open(args.pretrained_ofa_model, 'rb') as f:
    #     checkpoint = torch.load(f, map_location='cpu')
    checkpoint = load_weights(model, args.pretrained_ofa_model)
   
    # assert isinstance(checkpoint, dict)
    
    for k, v in model.state_dict().items():
        v.copy_(checkpoint.state_dict()[k])

    pareto_global = ofa_pruner.load_subnet_config('vww_pareto_global.txt')
    print(pareto_global.keys())
    exit()
    static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
        ofa_model, pareto_global[str(args.targeted_macs)]['subnet_setting'])

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