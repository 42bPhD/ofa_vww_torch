import torch
try:
    from .bfp_kernel import *
except:
    from bfp_kernel import *

def to_bfp(tensor, bit_width, block_size, out):
    if tensor.device.type == 'cuda':
        raise Exception("BFP operation only supports CPU, but got device type: " + tensor.device.type)
    CheckInputForBFP(tensor, bit_width, block_size)

    input = tensor.data_ptr()
    output = out.data_ptr()

    LaunchBFPCPUKernel(input, output, tensor.numel(), bit_width, block_size)
    return out

def to_bfp_v2(tensor, bit_width, block_size, out):
    if tensor.device.type == 'cuda':
        raise Exception("BFP operation only supports CPU, but got device type: " + tensor.device.type)
    CheckInputForBFPV2(tensor, bit_width, block_size)

    input = tensor.data_ptr()
    output = out.data_ptr()

    LaunchBFPCPUKernelV2(input, output, tensor.numel(), bit_width, block_size)
    return out

def to_bfp_prime_shared(tensor, bit_width, block_size, sub_block_size, sub_block_shift_bits, rounding_mode, out):
    if tensor.device.type == 'cuda':
        raise Exception("BFP operation only supports CPU, but got device type: " + tensor.device.type)
    CheckInputForBFPPrime(tensor, bit_width, block_size, sub_block_size)

    input = tensor.data_ptr()
    output = out.data_ptr()

    LaunchBFPPrimeCPUKernel(input, output, tensor.numel(), bit_width, block_size, sub_block_size, sub_block_shift_bits, rounding_mode)
    return out