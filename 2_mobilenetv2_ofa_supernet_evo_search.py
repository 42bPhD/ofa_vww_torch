# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import torch
import torchvision.datasets as datasets
from torchvision.models.mobilenet import mobilenet_v2
import torchvision.transforms as transforms
from pytorch_nndct import OFAPruner

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpus', type=str, default='0', help='String of available GPU number')
# ofa config
parser.add_argument(
    '--channel_divisible', type=int, default=8, help='make channel divisible')
parser.add_argument(
    '--expand_ratio', type=list, default=[0.25, 0.5, 0.75, 1], help='expand ratio')
parser.add_argument(
    '--excludes',
    type=list,
    default=None,
    help='excludes module')

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
    default=200,
    help='evo search targeted_min_macs')
parser.add_argument(
    '--evo_search_targeted_max_macs',
    type=int,
    default=250,
    help='evo search targeted_max_macs')

parser.add_argument(
    '--pretrained_ofa_model',
    type=str,
    default='model_best.pth.tar',
    help='Pretrained ofa model filepath')
parser.add_argument(
    '--data_dir',
    type=str,
    default="E:/1_TinyML/tiny/benchmark/training/visual_wake_words/vw_coco2014_96",
    help='Dataset directory')
parser.add_argument(
    '--num_workers',
    type=int,
    default=4,
    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

args, _ = parser.parse_known_args()

from utils.utils import get_gpus, eval_fn, calibration_fn


if __name__ == '__main__':

  gpus = get_gpus(args.gpus)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = mobilenet_v2(pretrained=True)

  traindir = os.path.join(args.data_dir, 'train')
  valdir = os.path.join(args.data_dir, 'validation')

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)

  val_dataset = datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ]))

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  model.to(device)

  criterion = torch.nn.CrossEntropyLoss().cuda()

  input_signature = torch.randn([1, 3, 224, 224], dtype=torch.float32).cuda()

  ofa_pruner = OFAPruner(model, input_signature)
  ofa_model = ofa_pruner.ofa_model(args.expand_ratio, args.channel_divisible,
                                   args.excludes)

  model = ofa_model.cuda()

  #reloading model
  with open(args.pretrained_ofa_model, 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')
  assert isinstance(checkpoint, dict)
  pretrained_state_dicts = checkpoint['state_dict']
  for k, v in model.state_dict().items():
    name = 'module.' + k if not k.startswith('module') else k
    v.copy_(pretrained_state_dicts[name])

  targeted_min_macs = args.evo_search_targeted_min_macs
  targeted_max_macs = args.evo_search_targeted_max_macs

  assert targeted_min_macs <= targeted_max_macs

  pareto_global = ofa_pruner.run_evolutionary_search(
      model, calibration_fn, (train_loader,), eval_fn, (val_loader,), 'acc@top1', 'max',
      targeted_min_macs, targeted_max_macs, args.evo_search_step,
      args.evo_search_parent_popu_size, args.evo_search_evo_iter,
      args.evo_search_mutate_size, args.evo_search_mutate_prob,
      args.evo_search_crossover_size)

  ofa_pruner.save_subnet_config(pareto_global, 'pareto_global.txt')
