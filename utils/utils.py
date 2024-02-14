import re
import torch
from torch import nn
from torch import optim
import shutil
import time
class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)

    def read(self):
        ret = self._desc
        self._desc = ''
        return ret

    def flush(self):
        pass

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

class Cross_entropy_loss_with_soft_target(torch.nn.modules.loss._Loss):
  def forward(self, output, target):
    target = torch.nn.functional.softmax(target, dim=1)
    logsoftmax = nn.functional.log_softmax
    return torch.mean(torch.sum(-target * logsoftmax(output, dim=1)))

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

def adjust_learning_rate(optimizer:optim, epoch:int, initial_lr:float):
    """adjust learning rate according to the epoch"""
    if epoch < 20:
        lr = initial_lr
    elif epoch < 30:  # 20 ~ 29 epochs
        lr = initial_lr / 2
    else:  # 30 ~ epochs
        lr = initial_lr / 4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate(optimizer, epoch, lr):
#     """Sets the learning rate to the initial LR decayed by every 2 epochs"""
#     lr = lr * (0.1**(epoch // 2))

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
        
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    shutil.copyfile(filename, 'model_best.pth')

def load_weights(model:nn.Module, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_gpus(device):
    return [int(i) for i in device.split(',')]

def eval_fn(model, dataloader_test):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader_test):
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            acc1, _ = accuracy(outputs, targets, topk=(1, 2))
            top1.update(acc1[0], images.size(0))
    return float(top1.avg)

def calibration_fn(model, train_loader, number_forward=16):
    model.eval()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.training = True
            m.momentum = None
            m.reset_running_stats()
    print("Calibration BN start...")
    with torch.no_grad():
        for index, (images, _) in enumerate(train_loader):
            images = images.cuda()
            model(images)
            if index > number_forward:
                break
    print("Calibration BN end...")
    
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