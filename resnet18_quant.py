import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from models.vww_model import mobilenet_v1

from tqdm import tqdm
from utils.utils import evaluate
from utils.dataloaders import get_subnet_dataloader, get_dataloader
from utils.io import load_weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',
                default="E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96",
                help="""Data set directory, when quant_mode=calib, it is for calibration, 
                        while quant_mode=test it is for evaluation""")
parser.add_argument('--model_dir',
                    default="E:/1_TinyML/embedded/trained_models/vww_baseline_912am2ws/vww_baseline_best.pth",
                    help="""Trained model file path. Download pretrained model from the following url 
                            and put it in model_dir specified path:
                            https://download.pytorch.org/models/resnet18-5c106cde.pth"""
)
parser.add_argument('--config_file',
                    default=None,
                    help='quantization configuration file')
parser.add_argument('--subset_len',
                    default=1000,
                    type=int,
                    help="""subset_len to evaluate model, 
                        using the whole validation dataset if it is not set""")
parser.add_argument('--batch_size',
                    default=50,
                    type=int,
                    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
                    default='calib', 
                    choices=['float', 'calib', 'test'], 
                    help="""quantization mode. 
                            0: no quantization, evaluate float model, 
                            calib: quantize, 
                            test: evaluate quantized model""")
parser.add_argument('--fast_finetune', 
                    dest='fast_finetune',
                    action='store_true',
                    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
                    dest='deploy',
                    action='store_true',
                    help='export xmodel for deployment')
parser.add_argument('--inspect', 
                    dest='inspect',
                    action='store_true',
                    help='inspect model')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Number of workers used in dataloading')
parser.add_argument("--image-size", 
                    default=96, 
                    type=int, 
                    help="Input image size (square assumed).")
parser.add_argument('--target', 
                    dest='target',
                    nargs="?",
                    const="",
                    help='specify target device')
args, _ = parser.parse_known_args()


  
# Extracted from the upper function 'evaluate'.
# In calibration, cannot evaluate the model accuracy because the quantization scales of tensors are kept being tuned.
def forward_loop(model, val_loader):
    model.eval()
    model = model.to(device)
    for iteraction, (images, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        outputs = model(images)

def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

    data_dir = args.data_dir
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    image_size = args.image_size
    inspect = args.inspect
    config_file = args.config_file
    target = args.target
    num_workers = args.num_workers
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    model = mobilenet_v1().cpu()
    model = load_weights(model, file_path)
    

    input = torch.randn([batch_size, 3, args.image_size, args.image_size])
    if quant_mode == 'float':
        quant_model = model
        if inspect:
            if not target:
                raise RuntimeError("A target should be specified for inspector.")
            import sys
            from pytorch_nndct.apis import Inspector
            # create inspector
            inspector = Inspector(target)  # by name
            # start to inspect
            inspector.inspect(quant_model, (input,), device=device)
            sys.exit()
      
    else:
        ####################################################################################
        # This function call will create a quantizer object and setup it. 
        # Eager mode model code will be converted to graph model. 
        # Quantization is not done here if it needs calibration.
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

        # Get the converted model to be quantized.
        quant_model = quantizer.quant_model
        
        #####################################################################################

    # to get loss value after evaluation
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    val_loader = get_subnet_dataloader(data_dir, 
                          subset_len, 
                          batch_size, 
                          image_size,
                          num_workers)
    
    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        ft_loader, _ = get_dataloader(data_dir, 
                                      batch_size, 
                                      image_size, 
                                      num_workers, 
                                      shuffle=True)

        if quant_mode == 'calib':
            quantizer.fast_finetune(forward_loop, (quant_model, ft_loader))
        elif quant_mode == 'test':
            quantizer.load_ft_param()
    
    if quant_mode == 'calib':
        # This function call is to do forward loop for model to be quantized.
        # Quantization calibration will be done after it.
        forward_loop(quant_model, val_loader)
        # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
        quantizer.export_quant_config()
    else:
        acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)
        # logging accuracy
        print('loss: %g' % (loss_gen))
        print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

    # handle quantization result
    if quant_mode == 'test' and  deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()


if __name__ == '__main__':

    model_name = 'mobilenet_v1'
    # file_path = os.path.join(args.model_dir, model_name + '.pth')
    file_path = args.model_dir

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(
        title=title,
        model_name=model_name,
        file_path=file_path)

    print("-------- End of {} test ".format(model_name))
