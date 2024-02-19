#!/usr/bin/env bash


import argparse
import os
import time

import torch
from pytorch_nndct import get_pruning_runner

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dataloaders import get_dataloader
from models.vww_model import mobilenet_v1
from utils import *
from pytorch_nndct import get_pruning_runner
from utils.utils import get_gpus
from utils.trains_evals import evaluate
from utils.io import load_weights

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', 
                    type=str, 
                    default='0',
                    help='String of available GPU number')
parser.add_argument('--sparsity',
                    type=float,
                    default=0.5,
                    help='Sparsity ratio')
parser.add_argument('--save_dir',
                    type=str,
                    default='./trained_models/iterative_pruned',
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
args, _ = parser.parse_known_args()

device = f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu'
gpus = get_gpus(args.gpus)



if __name__ == '__main__':
    sparse_model_path = os.path.join(args.save_dir, 'net_sparse.pth')
    assert os.path.exists(sparse_model_path), "No sparse model!"
    slim_model_path = os.path.join(args.save_dir, 'net_slim.pth')
    assert os.path.exists(sparse_model_path), "No slim model!"

    if not os.path.exists(args.data_dir):
        assert False, "No dataset found!"

        
    
    _ ,val_loader = get_dataloader(args.data_dir, 
                                    args.batch_size, 
                                    num_workers=args.num_workers, 
                                    shuffle=True, 
                                    image_size=args.image_size)

    sparse_model = mobilenet_v1()
    sparse_model = load_weights(sparse_model, sparse_model_path)
    
    input_signature = torch.randn([1, 3, args.image_size, args.image_size],
                                  dtype=torch.float32)
    input_signature = input_signature.to(device)
    sparse_model = sparse_model.to(device)
    pruning_runner = get_pruning_runner(sparse_model, input_signature, 'iterative')

    slim_model = pruning_runner.prune(removal_ratio=args.sparsity, mode='slim')
    slim_model = load_weights(slim_model, slim_model_path)
    slim_model = slim_model.to(device)
    sparse_model = sparse_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    acc1_sparse, acc5_sparse = evaluate(val_loader, sparse_model, criterion)
    print(f'Accuracy of sparse model: acc1={acc1_sparse:.4f}, acc5={acc5_sparse:.4f}')
    acc1_slim, acc5_slim = evaluate(val_loader, slim_model, criterion)
    print(f'Accuracy of slim model: acc1={acc1_slim:.4f}, acc5={acc5_slim:.4f}')
    assert acc1_sparse==acc1_slim and acc5_sparse==acc5_slim
    print('Done!')
