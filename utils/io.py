import os
import random
import string
import time
import torch
from typing import Optional, Union, List, Dict
import shutil
from torch import nn

import toml
import re
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

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    shutil.copyfile(filename, 'model_best.pth')

def load_weights(model:nn.Module, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

class TOMLConfig:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = self.load_config()
        self.resolve_references()

    def load_config(self):
        with open(self.filepath, 'r') as file:
            return toml.load(file)

    def resolve_references(self):
        # Iterate through all sections and keys, resolving references
        for section in self.config:
            for key in self.config[section]:
                self.config[section][key] = self.resolve_value(self.config[section][key], section)

    def resolve_value(self, value, current_section):
        # Ensure value is a string before attempting to resolve references
        if isinstance(value, str):
            pattern = r"\$\{([^}]+)\}"
            while True:
                match = re.search(pattern, value)
                if not match:
                    break
                ref_path = match.group(1)
                ref_value = self.get_ref_value(ref_path, current_section)
                value = value.replace(match.group(0), ref_value)
        return value

    def get_ref_value(self, ref_path, current_section):
        parts = ref_path.split('.')
        try:
            # Attempt to resolve reference from the current section or globally
            ref_value = self.config[current_section]
            for part in parts:
                if part in ref_value:
                    ref_value = ref_value[part]
                else:
                    # If not found in current section, try to resolve from the whole config
                    ref_value = self.config
                    for part in parts:
                        ref_value = ref_value[part]
            if isinstance(ref_value, dict):
                raise ValueError(f"Reference '{ref_path}' cannot be resolved to a string.")
            return str(ref_value)
        except KeyError:
            raise ValueError(f"Reference '{ref_path}' not found in configuration.")

    def __getitem__(self, item):
        return self.config[item]

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)
