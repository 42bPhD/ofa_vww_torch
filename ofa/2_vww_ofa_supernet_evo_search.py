
import argparse
import torch
import torchvision.datasets as datasets
import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2
import torchvision.transforms as transforms
from pytorch_nndct import OFAPruner
# from vww_model import mobilenet_v1 as mobilenet_v2
from utils.utils import get_gpus, calibration_fn, eval_fn, load_weights
from utils.dataloaders import get_dataloader
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=2, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes',
    type=list,
    default=None,
    help='excludes module')
parser.add_argument(
        "--image-size", default=96, type=int, help="Input image size (square assumed)."
    )
# evo_search config
parser.add_argument(
    '--evo_search_parent_popu_size',
    type=int,
    default=16,
    help='evo search parent popu size')
parser.add_argument(
    '--evo_search_mutate_size',
    type=int,
    default=8,
    help='evo search mutate size')
parser.add_argument(
    '--evo_search_crossover_size',
    type=int,
    default=4,
    help='evo search crossover size')
parser.add_argument(
    '--evo_search_mutate_prob',
    type=float,
    default=0.2,
    help='evo search mutate prob')
parser.add_argument(
    '--evo_search_evo_iter', type=int, default=10, help='evo search evo iter')
parser.add_argument(
    '--evo_search_step', type=int, default=10, help='evo search step')
parser.add_argument(
    '--evo_search_targeted_min_macs',
    type=int,
    default=32,
    help='evo search targeted_min_macs')
parser.add_argument(
    '--evo_search_targeted_max_macs',
    type=int,
    default=57,
    help='evo search targeted_max_macs')

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
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')

args, _ = parser.parse_known_args()


if __name__ == '__main__':
    import os
    gpus = get_gpus(args.gpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloader(dataset_dir=args.data_dir, 
                                            batch_size=args.batch_size, 
                                            image_size=args.image_size, 
                                            num_workers=args.workers, 
                                            shuffle=True)
    model = mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, 2)
    # model = load_weights(model, './vww_trained.pth')

    
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    input_signature = torch.randn([1, 3, args.image_size, args.image_size], dtype=torch.float32).cuda()

    ofa_pruner = OFAPruner(model, input_signature)
    ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                    args.excludes)

    model = ofa_model.cuda()
    #reloading model
    checkpoint = load_weights(model, args.pretrained_ofa_model)
    
    
    # assert isinstance(checkpoint, dict)
    for k, v in model.state_dict().items():
        v.copy_(checkpoint.state_dict()[k])


    dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
        model, 'max')
    static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
        dynamic_subnet, dynamic_subnet_setting)

    max_subnet_macs = macs

    dynamic_subnet, dynamic_subnet_setting = ofa_pruner.sample_subnet(
        model, 'min')
    static_subnet, _, macs, params = ofa_pruner.get_static_subnet(
        dynamic_subnet, dynamic_subnet_setting)

    min_subnet_macs = macs

    print('max subnet macs(M):', max_subnet_macs)
    print('min subnet macs(M):', min_subnet_macs)

    targeted_min_macs = args.evo_search_targeted_min_macs
    targeted_max_macs = args.evo_search_targeted_max_macs
        
    if targeted_max_macs > max_subnet_macs:
        targeted_max_macs = max_subnet_macs-1
        
    if min_subnet_macs > targeted_min_macs: #74 > 200
        targeted_min_macs = min_subnet_macs+1

    if targeted_min_macs > targeted_max_macs:
        targeted_min_macs = max_subnet_macs - (max_subnet_macs - min_subnet_macs)
        
    assert targeted_min_macs <= targeted_max_macs and min_subnet_macs <= targeted_min_macs and targeted_max_macs <= max_subnet_macs
    print('Modified Targeted macs range:', targeted_min_macs, targeted_max_macs)
    pareto_global = ofa_pruner.run_evolutionary_search(
        model, calibration_fn, (train_loader,), eval_fn, (val_loader,), 'acc@top1', 'max',
        targeted_min_macs, targeted_max_macs, args.evo_search_step,
        args.evo_search_parent_popu_size, args.evo_search_evo_iter,
        args.evo_search_mutate_size, args.evo_search_mutate_prob,
        args.evo_search_crossover_size)

    ofa_pruner.save_subnet_config(pareto_global, 'vww_pareto_global.txt')