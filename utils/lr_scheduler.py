import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        last_epoch: int = -1,
    ):
        """Scheduler for learning rate warmup.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer, e.g. SGD.
        total_iters: int
            Number of iterations for warmup Learning rate phase.
        last_epoch: int
            The index of last epoch. Default: -1
        """
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return current learning rate."""
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]