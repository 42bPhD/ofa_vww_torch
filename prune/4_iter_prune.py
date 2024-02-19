#!/usr/bin/env bash

import argparse
import os

import torch
from pytorch_nndct import get_pruning_runner

import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dataloaders import get_dataloader, get_subnet_dataloader
from models.vww_model import mobilenet_v1
from utils import *
from pytorch_nndct import get_pruning_runner
from utils.utils import eval_fn, get_gpus
from utils.io import load_weights, save_checkpoint, StepMonitor



parser = argparse.ArgumentParser()
parser.add_argument('--sparsity',
                    type=float,
                    default=0.5,
                    help='Sparsity ratio')
parser.add_argument('--save_dir',
                    type=str,
                    default='./trained_models/iter_pruned',
                    help='Where to save retrained model')
parser.add_argument("--image-size", 
                    default=96, 
                    type=int, 
                    help="Input image size (square assumed).")
args, _ = parser.parse_known_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



if __name__ == '__main__':
    sparse_model_path = os.path.join(args.save_dir, 'net_sparse.pth')
    assert os.path.exists(sparse_model_path), "No sparse model!"
    step_Monitor = StepMonitor(model_name = args.save_dir)
    filename = 'net_slim.pth'
    slim_model_path = os.path.join(step_Monitor.version,filename)
    
    sparse_model = mobilenet_v1()
    sparse_model = load_weights(sparse_model, sparse_model_path)
    sparse_model.to(device)
    input_signature = torch.randn([1, 3, args.image_size, args.image_size], 
                                  dtype=torch.float32)
    input_signature = input_signature.to(device)

    pruning_runner = get_pruning_runner(sparse_model, 
                                        input_signature, 
                                        'iterative')
    slim_model = pruning_runner.prune(removal_ratio=args.sparsity,
                                      mode='slim')
    from torchsummary import summary
    summary(slim_model, (3, 96, 96))
    
    save_checkpoint(state = {
                            'state_dict': slim_model.cpu().state_dict()
                            }, 
                    filename=slim_model_path)
    print('Convert sparse model to slim model done!')
