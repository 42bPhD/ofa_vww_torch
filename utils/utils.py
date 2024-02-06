import re
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

from torch import optim
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
