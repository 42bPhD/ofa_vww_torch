#!/usr/bin/env bash
import argparse
import os
import time
import torch
from pytorch_nndct import get_pruning_runner


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import eval_fn, get_gpus, calibration_fn
from utils.io import load_weights
from utils.dataloaders import get_dataloader, get_subnet_dataloader
from models.vww_model import mobilenet_v1


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', 
                    type=str, 
                    default='0',
                    help='String of available GPU number')
parser.add_argument('--subset_len',
                    default=None,
                    help='Subset length for evaluating model in analysis, using the whole validation dataset if it is not set')
parser.add_argument('--num_subnet',
                    type=int,
                    default=40,
                    help='The number of subnets searched')
parser.add_argument('--sparsity',
                    type=float,
                    default=0.5,
                    help='Sparsity ratio')
parser.add_argument('--pretrained',
                    type=str,
                    default='E:\\1_TinyML\\embedded\\trained_models\\vww_baseline_912am2ws\\vww_baseline_best.pth',
                    help='Pretrained model filepath')
parser.add_argument('--save_dir',
                    type=str,
                    default='./trained_models/one_step_pruned',
                    help='Where to save retrained model')
parser.add_argument('--data_dir',
                    type=str,
                    default='E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96',
                    help='Dataset directory')
parser.add_argument("--image-size", 
                    default=96, 
                    type=int, 
                    help="Input image size (square assumed).")
parser.add_argument('--num_workers',
                    type=int,
                    default=4,
                    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=50, 
                    help='Batch size')
args, _ = parser.parse_known_args()

device = f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu'
gpus = get_gpus(args.gpus)


if __name__ == '__main__':
    assert os.path.exists(args.pretrained), "No pretrained model!"
    if not os.path.exists(args.data_dir):
        assert False, "No dataset found!"
  
    if args.subset_len:
        data_loader = get_subnet_dataloader(args.data_dir, 
                                            args.batch_size, 
                                            args.subnet_len, 
                                            num_workers=args.num_workers,
                                            shuffle=False)
    else:
        _, data_loader = get_dataloader(args.data_dir, 
                                        args.batch_size, 
                                        num_workers=args.num_workers, 
                                        shuffle=False)

    model = mobilenet_v1()
    model = load_weights(model, args.pretrained)
    input_signature = torch.randn([1, 3, args.image_size, args.image_size], 
                                  dtype=torch.float32)
    input_signature = input_signature.to(device)
    model = model.to(device)
    pruning_runner = get_pruning_runner(model, input_signature, 'one_step')

    pruning_runner.search(
        gpus=gpus,
        calibration_fn=calibration_fn,
        calib_args=(data_loader,),
        num_subnet=args.num_subnet,
        removal_ratio=args.sparsity,
        eval_fn=eval_fn,
        eval_args=(data_loader,)) 