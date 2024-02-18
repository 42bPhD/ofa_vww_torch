import os
import random
import string
import time
import torch
from typing import Optional, Union, List, Dict
import shutil
from torch import nn
class StepMonitor(object):
    def __init__(
        self,
        model_name:str,
    ):
        """Custom checkpointer class that stores checkpoints in an easier to access way.

        Args:
            cfg (tomllib): DictConfig containing at least an attribute name.
        """

        # 'model_name'+random_string
        self.model_name = model_name
        
        self.version = f"{self.model_name}_{self.random_string()}"
        #Check Empty Directory
        os.makedirs(self.version, exist_ok=True)
        
    @staticmethod
    def random_string(letter_count=4, digit_count=4):
        tmp_random = random.Random(time.time())
        rand_str = "".join(tmp_random.choice(string.ascii_lowercase) for _ in range(letter_count))
        rand_str += "".join(tmp_random.choice(string.digits) for _ in range(digit_count))
        rand_str = list(rand_str)
        tmp_random.shuffle(rand_str)
        return "".join(rand_str)

    def save_checkpoint(self, state, filename='checkpoint.pth', is_best=False):
        torch.save(state, os.path.join(self.version, filename))
        if is_best:
            shutil.copyfile(os.path.join(self.version, filename), 'model_best.pth')

    def load_weights(self, model:nn.Module, model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model