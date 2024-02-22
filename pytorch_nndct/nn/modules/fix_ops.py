

#
# Copyright 2019 Xilinx Inc.
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
#


# import os
import sys
import torch

from pysrc.nndct_fix_kernels import *
from pysrc.nndct_fix_kernels import _fix_neuron_v2_cpu
from pysrc.nndct_math_cpu import *
from pysrc.nndct_fixneuron_op import *
from pysrc.nndct_diffs_op import *
from pysrc.bfp_kernel import *
from pysrc.fix_kernel import *


import copy
import numpy as np
from nndct_shared.utils import NndctOption, NndctScreenLogger
from pytorch_nndct.nn.utils.decorator import pre_and_post_process_f16_tensor
from pytorch_nndct.utils.torch_utils import CmpFlag, compare_torch_version
# from torch.utils.cpp_extension import load
__all__ = ["NndctFixNeuron",
           "NndctRound", \
           "NndctDiffsFixPos",\
           "NndctDiffsFixPosChannel",\
           "NndctSigmoidTableLookup",\
           "NndctSigmoidTableLookupAIE2",\
           "NndctSigmoidSimulation",\
           "NndctTanhTableLookup",\
           "NndctTanhTableLookupAIE2",\
           "NndctTanhSimulation", \
           "FixNeuronWithBackward",\
           "fake_quantize_per_tensor",\
           "NndctSoftmaxExpApproximate",\
           "NndctSoftmaxLOD",\
           "NndctSoftmaxSimulationPart1",\
           "NndctSoftmaxSimulationPart2",\
           "fake_quantize_per_channel",\
           "fake_quantize_per_tensor_tensorrt",\
           "fake_quantize_per_channel_tensorrt",\
           "NndctExpApprAIE2",\
           "NndctInverseAIE2",\
           "NndctLogSoftmaxFastLn",\
           "NndctLogSoftmaxSub",\
           "NndctAIESqrt",\
           "NndctAIEISqrt",\
           "NndctISqrt",\
           "NndctLayernormInvSqrt"]     

def support_onnx_export():
  if compare_torch_version(CmpFlag.GREATER_EQUAL, "1.7.0"):
    return True
  else:
    return False

def clone_view_tensor(tensor):
  cloned_tensor = tensor
  if (isinstance(tensor, torch.Tensor) and
   hasattr(tensor, "storage")  and 
   hasattr(tensor, "numel") and 
   tensor.untyped_storage().size() != tensor.numel()):
    cloned_tensor = tensor.clone()
  return cloned_tensor

class FixNeuronFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, scale_inv, zero_point, quant_max, method=2):
    ctx.save_for_backward(input)
    return fake_quantize_per_tensor(input, scale_inv, zero_point, -quant_max, quant_max-1, method)
  @staticmethod
  def backward(ctx, grad_output):
    grad_input = grad_scale_inv = grad_zero_point = grad_quant_max = grad_method = None
    if ctx.needs_input_grad[0]:
      grad_input = grad_output
    return grad_input, grad_scale_inv, grad_zero_point, grad_quant_max, grad_method

class FixNeuronWithBackward(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()

  def forward(self, input, scale_inv, zero_point, quant_max, method=2):
    output = FixNeuronFunc.apply(input, scale_inv, zero_point, quant_max, method)
    return output

@pre_and_post_process_f16_tensor
def NndctRound(Tinput, Toutput, method=2):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Toutput = Round(Tinput, Toutput, method, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctFixNeuron(Tinput, Toutput, maxamp, method=2):
  valmax, valamp = maxamp[0], maxamp[1]
  valmin = -valmax
  valmax = valmax - 1
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if support_onnx_export():
    Toutput = fix_neuron(Tinput, valmin, valmax, 
                        valamp, 0, method, device_id, 1)
  else:
    Toutput = FixNeuronV2(Tinput, Toutput, valmin, valmax, 
                          valamp, 0, method, device_id)
  return Toutput
  '''
  if Tinput.device == torch.device("cpu"):
    output = Tinput.cuda()
    nndct_kernels.FixNeuronV2(output, output, valmax,
                              valamp, method)
    Tinput.copy_(output.cpu())
    return Tinput

    # cpu fix neuron
    """
    # output = Tinput.cpu().detach().numpy()
    # output = output * valamp
    # if method == 2:
    #   output = np.where(output > valmax - 1, (valmax - 1), output)
    #   output = np.where(output < (-valmax), -valmax, output)
    #   output = np.where(np.logical_and(output > 0, np.logical_and(np.floor(output) % 2 == 0, output - np.floor(output) == 0.5)), np.ceil(output), output)
    #   output = np.where(output >= 0, np.round(output), output)
    #   output = np.where(np.logical_and(output < 0, output - np.floor(output) == 0.5), np.ceil(output), output)
    #   output = np.where(output < 0, np.round(output), output)

    # elif method == 3:
    #   output = np.where(output > valmax - 1, (valmax - 1), output)
    #   output = np.where(output < (-valmax), -valmax, output)
    #   output = np.where(np.logical_and(output > 0, np.logical_and(np.floor(output) % 2 == 0, output - np.floor(output) == 0.5)), np.ceil(output), output)
    #   output = np.where(output >= 0, np.round(output), output)
    #   output = np.where(np.logical_and(output < 0, np.logical_and(np.ceil(output) % 2 == 0, output - np.floor(output) == 0.5)), np.floor(output), output)
    #   output = np.where(output < 0, np.round(output), output)

    # Tinput.copy_(torch.from_numpy(output))
    # Tinput.div_(valamp)
    # return Tinput
    """
  else:
    nndct_kernels.FixNeuronV2(Tinput, Toutput, valmax,
                              valamp, method)
  return Toutput
  '''

@pre_and_post_process_f16_tensor
def NndctDiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width=8, range=5, method=2):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if support_onnx_export():
    Tbuffer = diffs_fix_pos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)
  else:
    Tbuffer = DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)
  return Tbuffer

@pre_and_post_process_f16_tensor
def NndctDiffsFixPosChannel(Tinput, Tbuffer, Tfixpos, axis, bit_width=8, scope=5, method=2):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  input_split = torch.split(Tinput, 1, dim=axis)
  buffer_split = torch.split(Tbuffer, 1, dim=axis)
  # TODO(@kewang): The split is a tensor view operation. Is it neccessary to clone tensor before calib and test ? 
  if support_onnx_export():
    for i in range(len(input_split)):
      buffer_split[i] = diffs_fix_pos(input_split[i], buffer_split[i], Tfixpos[i], bit_width, scope, method, device_id)
  else:
    for i in range(len(input_split)):
      buffer_split[i] = DiffsFixPos(input_split[i], buffer_split[i], Tfixpos[i], bit_width, scope, method, device_id)
  return buffer_split

@pre_and_post_process_f16_tensor
def NndctSigmoidTableLookup(Tinput, Ttable, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if support_onnx_export():
    Toutput = SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
  else:
    Toutput = SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctSigmoidSimulation(Tinput, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if device_id == 1:
    print("Sigmoid simulation does not support CPU")
  else:
    if support_onnx_export():
      Toutput = SigmoidSimulation(Tinput, Toutput, fragpos, device_id)
    else:
      Toutput = SigmoidSimulation(Tinput, Toutput, fragpos, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctTanhTableLookup(Tinput, Ttable, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if support_onnx_export():
    Toutput = TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
  else:
    Toutput = TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctTanhSimulation(Tinput, Toutput, fragpos):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if device_id == 1:
    print("Tanh simulation does not support CPU")
  else:
    if support_onnx_export():
      Toutput = TanhSimulation(Tinput, Toutput, fragpos, device_id)
    else:
      Toutput = TanhSimulation(Tinput, Toutput, fragpos, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctSoftmaxExpApproximate(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if device_id == 1:
    print("Softmax Exponent Approximate does not support CPU")
  else:
    if support_onnx_export():
      Toutput = SoftmaxExpApproximate(Tinput, Toutput, device_id)
    else:
      Toutput = SoftmaxExpApproximate(Tinput, Toutput, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctSoftmaxLOD(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if device_id == 1:
    print("Softmax LOD does not support CPU")
  else:
    if support_onnx_export():
      Toutput = SoftmaxLOD(Tinput, Toutput, device_id)
    else:
      Toutput = SoftmaxLOD(Tinput, Toutput, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctSoftmaxSimulationPart1(Tinput, Toutput):
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Tinput = clone_view_tensor(Tinput)
  if device_id == 1:
    print("Softmax Simulation Part 1 does not support CPU")
  else:
    if support_onnx_export():
      Toutput = SoftmaxSimulationPart1(Tinput, Toutput, device_id)
    else:
      Toutput = SoftmaxSimulationPart1(Tinput, Toutput, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctSoftmaxSimulationPart2(sum, Toutput):
  device_id = 1 if Toutput.device == torch.device("cpu") else 0
  sum = clone_view_tensor(sum)
  if device_id == 1:
    print("Softmax Simulation Part 2 does not support CPU")
  else:
    if support_onnx_export():
      Toutput = SoftmaxSimulationPart2(sum, Toutput, device_id)
    else:
      Toutput = SoftmaxSimulationPart2(sum, Toutput, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def fake_quantize_per_tensor(input, scale_inv, zero_point, quant_min, quant_max, method, inplace):
  if method == -1:
    scale_inv = scale_inv.item() if isinstance(scale_inv, torch.Tensor) else scale_inv
    if not inplace:
      return torch.fake_quantize_per_tensor_affine(input, 1.0 / scale_inv, zero_point, quant_min, quant_max)
    else:
      out = torch.fake_quantize_per_tensor_affine(input, 1.0 / scale_inv, zero_point, quant_min, quant_max)
      input.data.copy_(out.data)
      return input
  else:
    input = clone_view_tensor(input)
    device_id = 1 if input.device == torch.device("cpu") else 0
   
    if support_onnx_export():
      output = fix_neuron(input, quant_min, quant_max, 
                                        scale_inv, zero_point, method, 
                                        device_id, inplace)
      return output
    else:
      output = input.clone() if inplace == 0 else input
      output = FixNeuronV2(input, output, quant_min,
                           quant_max, scale_inv, zero_point, 
                           method, device_id)
      return output

@pre_and_post_process_f16_tensor
def fake_quantize_per_channel(input, scale_inv, zero_point, axis, quant_min, quant_max, method, inplace):
  if method == -1:
    if compare_torch_version(CmpFlag.GREATER, "0.9.0"):
      zero_point = zero_point.to(torch.int32)
    else:
      zero_point = zero_point.to(torch.long)
    return torch.fake_quantize_per_channel_affine(input, 1.0 / scale_inv, zero_point, axis, quant_min, quant_max)
  else:
    device_id = 1 if input.device == torch.device("cpu") else 0
    if support_onnx_export():
      scale = torch.where(scale_inv<sys.float_info.min, torch.tensor(sys.float_info.max, dtype=scale_inv.dtype, device=scale_inv.device), 1.0/scale_inv).to(torch.float)
      # api: (tensor(float), int32, int32, tensor(float), tensor(int8), int32, int32, int32, bool)
      output = fix_neuron_per_channel(input, quant_min, quant_max, scale, zero_point.to(torch.int8), axis, method, device_id, inplace)
      return output
    else:
      input_split = torch.split(input, 1, dim=axis)
      output_cat = []
      for i in range(len(input_split)):
        output_split = input_split[i].clone() if inplace == 0 else input_split[i]
        output_split = FixNeuronV2(input_split[i], output_split, quant_min,
                    quant_max, scale_inv[i], zero_point[i], 
                    method, device_id)
        output_cat.append(output_split)
      output = torch.cat(output_cat, axis)
      return output
  
@pre_and_post_process_f16_tensor
def fake_quantize_per_channel_tensorrt(inputs, amax, min_bound, max_bound, axis=None):
  # Computation must be in FP32 to prevent potential over flow.
  if not isinstance(max_bound, torch.Tensor):
    max_bound = torch.tensor(float(max_bound))
  #max_bound = max_bound.double()
  
  input_dtype = inputs.dtype
  if inputs.dtype == torch.half:
    inputs = inputs.float()

  min_amax = amax.min()
  if min_amax < 0:
    raise ValueError("Negative values in amax")

  scale = max_bound / amax

  epsilon = 1. / (1<<24)
  if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    zero_amax_mask = (amax <= epsilon)
    scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0
  
  if axis != None:
    for x_dim in range(inputs.ndim):
      if x_dim != axis:
        scale = torch.unsqueeze(scale, x_dim)

  outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

  if min_amax <= epsilon:
    scale[zero_amax_mask] = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

  if input_dtype == torch.half:
    outputs = outputs.half()
  outputs = outputs / scale
  return outputs

@pre_and_post_process_f16_tensor
def fake_quantize_per_tensor_tensorrt(inputs, amax, min_bound, max_bound):
  # Computation must be in FP32 to prevent potential over flow.
  if not isinstance(max_bound, torch.Tensor):
    max_bound = torch.tensor(float(max_bound))
  #max_bound = max_bound.double()
  
  if not isinstance(amax, torch.Tensor):
    amax = torch.tensor(float(amax))
  #amax = amax.double()
  
  input_dtype = inputs.dtype
  if inputs.dtype == torch.half:
    inputs = inputs.float()

  if amax < 0:
    raise ValueError("Negative values in amax")

  scale = max_bound / amax
  epsilon = 1. / (1<<24)
  if amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
    scale = 0  # Value quantized with amax=0 should all be 0

  outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

  if amax <= epsilon:
    scale = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

  if input_dtype == torch.half:
    outputs = outputs.half()
  
  outputs = outputs / scale
  return outputs

# def fake_quantize_tensorrt(inputs, amax, num_bits=8, unsigned=False, narrow_range=True):
#     """Shared function body between TensorQuantFunction and FakeTensorQuantFunction"""
#     # Fine scale, per channel scale will be handled by broadcasting, which could be tricky. Pop a warning.
#     if isinstance(amax, torch.Tensor) and inputs.dim() != amax.dim():
#       logging.debug("amax %s has different shape than inputs %s. Make sure broadcast works as expected!",
#                     amax.size(), inputs.size())

#     if unsigned:
#       if inputs.min() < 0.:
#         raise TypeError("Negative values encountered in unsigned quantization.")

#     # Computation must be in FP32 to prevent potential over flow.
#     input_dtype = inputs.dtype
#     if inputs.dtype == torch.half:
#       inputs = inputs.float()
#     if amax.dtype == torch.half:
#       amax = amax.float()

#     min_amax = amax.min()
#     if min_amax < 0:
#       raise ValueError("Negative values in amax")

#     max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
#     if unsigned:
#       min_bound = 0
#     elif narrow_range:
#       min_bound = -max_bound
#     else:
#       min_bound = -max_bound - 1
#     scale = max_bound / amax

#     epsilon = 1. / (1<<24)
#     if min_amax <= epsilon:  # Treat amax smaller than minimum representable of fp16 0
#       zero_amax_mask = (amax <= epsilon)
#       scale[zero_amax_mask] = 0  # Value quantized with amax=0 should all be 0

#     outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)

#     if min_amax <= epsilon:
#       scale[zero_amax_mask] = 1.  # Return 1 makes more sense for values quantized to 0 with amax=0

#     if input_dtype == torch.half:
#       outputs = outputs.half()

#     return outputs, scale

@pre_and_post_process_f16_tensor
def NndctSigmoidTableLookupAIE2(Tinput, Toutput, fragpos):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Toutput = SigmoidTableLookupAIE2(Tinput, Toutput, fragpos, device_id)
  return Toutput
  

@pre_and_post_process_f16_tensor
def NndctTanhTableLookupAIE2(Tinput, Toutput, fragpos):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Toutput = TanhTableLookupAIE2(Tinput, Toutput, fragpos, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctExpApprAIE2(Tinput, Toutput, bit_width):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Exp Approximation does not support CPU")
  else:
    Toutput = ExpApprAIE2(Tinput, Toutput, device_id, bit_width)
    return Toutput

@pre_and_post_process_f16_tensor
def NndctLogSoftmaxFastLn(Tinput, Toutput):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("LogSoftmax fast ln does not support CPU")
  else:
    Toutput = LogSoftmaxFastLn(Tinput, Toutput, device_id)
    return Toutput

@pre_and_post_process_f16_tensor
def NndctLogSoftmaxSub(Tinput, Toutput, Tsum):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("LogSoftmax subtraction does not support CPU")
  else:
    Toutput = LogSoftmaxSub(Tinput, Toutput, Tsum, device_id)
    return Toutput

@pre_and_post_process_f16_tensor
def NndctAIESqrt(Tinput, Toutput):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Toutput = AIESqrt(Tinput, Toutput, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctAIEISqrt(Tinput, Toutput):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Toutput = AIEISqrt(Tinput, Toutput, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctISqrt(Tinput, Toutput):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  Toutput = LayernormISqrt(Tinput, Toutput, device_id)
  return Toutput

@pre_and_post_process_f16_tensor
def NndctLayernormInvSqrt(Tinput, Toutput):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Layernorm InvSqrt does not support CPU")
  else:
    Toutput = LayernormInvSqrt(Tinput, Toutput, device_id)
    return Toutput

@pre_and_post_process_f16_tensor
def NndctInverseAIE2(Tinput, Toutput):
  Tinput = clone_view_tensor(Tinput)
  device_id = 1 if Tinput.device == torch.device("cpu") else 0
  if device_id == 1:
    print("Inverse AIE2 does not support CPU")
  else:
    Toutput = InverseAIE2(Tinput, Toutput, device_id)
    return Toutput
    
def diffs_fix_pos(input, bit_width, scope, method):
    # get max and min element in the tensor
    abs_max = 1 << (bit_width - 1)
    fix_lb = -abs_max - 0.5
    fix_ub = abs_max - 0.5
    x_max = torch.max(input)
    x_min = torch.min(input)

    # calculate step and fix pos based on max and min value
    step = torch.max(x_min / fix_lb, x_max / fix_ub)
    max_scale = torch.floor(torch.log2(1.0/step)) if step > sys.float_info.min else torch.tensor(18)

    # calculate step based on diffs 
    final_scale = max_scale
    fixed_diff_min = sys.float_info.max
    if scope > 1:
      # avoid clone multiple times
      input = clone_view_tensor(input)
      for i in range(0, scope):
        scale = max_scale + i
        qinput = fake_quantize_per_tensor(
            input,
            pow(2.0, scale), 
            0, 
            -abs_max, 
            abs_max-1, 
            method, 
            0)
        qinput = torch.sub(input, qinput)
        qinput = torch.pow(qinput, 2.0)
        diff = torch.sum(qinput).item()
        if diff < fixed_diff_min:
          final_scale = scale
          fixed_diff_min = diff

    return final_scale
