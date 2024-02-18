
import argparse
import time
import torch
from pytorch_nndct import OFAPruner

import argparse
import time
import torch
from torchvision.models.mobilenet import mobilenet_v2
from pytorch_nndct import OFAPruner
from utils.utils import get_gpus, AverageMeter, ProgressMeter, accuracy, evaluate

from utils.utils import get_gpus, load_weights
from utils.dataloaders import get_dataloader
from models.vww_model import mobilenet_v1 as mobilenet_v2
import os

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
    default=40,
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



import numpy as np
from torch import nn
import onnxruntime
from pytorch_nndct.expanding.expanding_lib import expand_and_export, load_expanded_model

def do_expand_and_export(model, model_name, input_signature, channel_divisibles, out_dir):
    expand_and_export(model_name, model, input_signature, channel_divisibles, out_dir, 
        onnx_export_kwargs= {
        "input_names": ['input'],
        "output_names": ['output'],
        "dynamic_axes": {'input' : {0 : 'batch_size'},  
                            'output' : {0 : 'batch_size'}}
        })

def load_expanded_torch_model(model_name: str, model: nn.Module, input_signature: torch.Tensor, dir_path: str):
    expanding_spec_path = os.path.join(dir_path, "expanding_spec")
    model_path = os.path.join(dir_path, model_name + ".pth")
    return load_expanded_model(expanding_spec_path, model, input_signature, model_path)

def load_expanded_onnx_model(model_name: str, dir_path: str) -> onnxruntime.InferenceSession:
    model_path = os.path.join(dir_path, model_name + "_fp32.onnx")
    return onnxruntime.InferenceSession(model_path)

def summary(channel_divisible: int, raw_output: np.ndarray, torch_output: np.ndarray, onnx_output: np.ndarray) -> None:
    raw_torch_diff = abs(raw_output - torch_output)
    raw_torch_relative_diff = raw_torch_diff / np.concatenate((abs(raw_output), abs(torch_output)), axis=0).max(axis=0)
    raw_onnx_diff = abs(raw_output - onnx_output)
    raw_onnx_relative_diff = raw_onnx_diff / np.concatenate((abs(raw_output), abs(onnx_output)), axis=0).max(axis=0)
    print("relative diff for channel_divisible {}".format(channel_divisible))
    print("row-torch relative diff: max = {:.4f}, min = {:.4f}, average = {:.4f}"
        .format(raw_torch_relative_diff.max(), raw_torch_relative_diff.min(), raw_torch_relative_diff.mean()))
    print("row-onnx relative diff: max = {:.4f}, min = {:.4f}, average = {:.4f}"
        .format(raw_onnx_relative_diff.max(), raw_onnx_relative_diff.min(), raw_onnx_relative_diff.mean()))

    

def verify(model_name: str, model: nn.Module, input_signature: torch.Tensor, out_dir: str, channel_divisibles: list) -> None:
    for channel_divisible in channel_divisibles:
        dir_path = os.path.join(out_dir, model_name + "_padded_{}".format(channel_divisible))
        torch_model = load_expanded_torch_model(model_name, model, input_signature, dir_path).eval()
        onnx_model = load_expanded_onnx_model(model_name, dir_path)
        raw_output = model(input_signature).detach().cpu().numpy()
        torch_output = torch_model(input_signature).detach().cpu().numpy()
        onnx_output = onnx_model.run(None, {"input": input_signature.detach().cpu().numpy()})[0]
        summary(channel_divisible, raw_output, torch_output, onnx_output)
if __name__ == '__main__':
    gpus = get_gpus(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, 2)
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
    
    out_dir = "./trained_models/"
    model_name = "mobilenet_v2"
    do_expand_and_export(model, model_name, input_signature, [args.channel_divisible], out_dir)
    verify(model_name, model, input_signature, out_dir, [args.channel_divisible])
