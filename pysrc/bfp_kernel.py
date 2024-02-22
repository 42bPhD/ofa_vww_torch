import numpy as np
import torch

def __float_as_uint(x):
    return np.frombuffer(x, dtype=np.uint32)

def __uint_as_float(x):
    return np.frombuffer(x, dtype=np.float32)

def CheckInputForBFP(tensor, bit_width, block_size):
    assert tensor.is_contiguous(), "Input tensor must be contiguous."
    assert tensor.dtype == torch.float32, "Tensor with dtype float32 can be quantized to BFP, but got {}".format(tensor.dtype)
    assert tensor.numel() % block_size == 0, "The number of elements of tensor must be divisible by 'block_size'"
    assert 10 <= bit_width <= 16, "Bitwidth must be in [10, 16]"

def CheckInputForBFPV2(tensor, bit_width, block_size):
    CheckInputForBFP(tensor, bit_width, block_size)
    axis_size = tensor.size(tensor.dim() - 1)
    assert axis_size % block_size == 0, "The number of elements in last axis must be divisible by 'block_size'"

def CheckInputForBFPPrime(tensor, bit_width, block_size, sub_block_size):
    CheckInputForBFPV2(tensor, bit_width, block_size)
    assert block_size % sub_block_size == 0, "The 'block_size' must be divisible by 'sub_block_size'"

def GetExponentCPU(v):
    uint_v = __float_as_uint(v)
    return (uint_v & 0x7f800000) >> 23

def GetMaxExponentCPU(input, n):
    max_exp = 0
    for i in range(n):
        max_exp = max(max_exp, GetExponentCPU(input[i]))
    return max_exp

def BFPCPUKernel(input, output, n, index, stride, bit_width):
    shared_exp = 0
    for i in range(index, n, stride):
        exp = GetExponentCPU(input[i])
        if exp == 0xff:
            exp = 0
        if exp > shared_exp:
            shared_exp = exp

    shared_exp_value = int(shared_exp) - 127
    m_bits = bit_width - 9
    scale = 2.0 ** (shared_exp_value - (m_bits - 1))
    max_v = 2.0 ** (shared_exp_value + 1) - scale
    for i in range(index, n, stride):
        exp = GetExponentCPU(input[i])
        if exp == 0xff:
            output[i] = input[i]
        else:
            x = round(input[i] / scale) * scale
            output[i] = max(-max_v, min(x, max_v))

def LaunchBFPCPUKernel(input, output, n, bit_width, block_size):
    num_blocks = n // block_size
    for index in range(num_blocks):
        BFPCPUKernel(input, output, n, index, num_blocks, bit_width)
        
def BFPCPUKernelV2(input, output, offset, bit_width, block_size):
    shared_exp = 0
    for i in range(block_size):
        exp = GetExponentCPU(input[offset + i])
        if exp == 0xff:
            exp = 0
        if exp > shared_exp:
            shared_exp = exp

    shared_exp_value = int(shared_exp) - 127
    m_bits = bit_width - 9
    scale = 2.0 ** (shared_exp_value - (m_bits - 1))
    max_v = 2.0 ** (shared_exp_value + 1) - scale

    for i in range(block_size):
        exp = GetExponentCPU(input[offset + i])
        if exp == 0xff:
            output[i] = input[i]
        else:
            x = round(input[i] / scale) * scale
            output[i] = max(-max_v, min(x, max_v))

def LaunchBFPCPUKernelV2(input, output, n, bit_width, block_size):
    num_blocks = n // block_size
    for index in range(num_blocks):
        BFPCPUKernel(input, output, index * block_size + block_size, index * block_size, 1, bit_width)

def LaunchBFPPrimeCPUKernel(input, output, n, bit_width, block_size, sub_block_size, sub_block_shift_bits, rounding_mode):
    num_blocks = n // block_size
    for index in range(num_blocks):
        BFPPrimeCPUKernel(input, output, n, index * block_size, 1, bit_width, block_size, sub_block_size, sub_block_shift_bits, rounding_mode)

def BFPPrimeCPUKernel(input, output, n, offset, stride, bit_width, block_size, sub_block_size, sub_block_shift_bits, rounding_mode):
    m_float = 23
    m_bfp = bit_width - 9
    exp_bias = 127

    shared_exp = GetMaxExponentCPU(input + offset, block_size)

    for i in range(block_size // sub_block_size):
        max_sub_exp = GetMaxExponentCPU(input + offset + i * sub_block_size, sub_block_size)

        shift_upper_bound = (1 << sub_block_shift_bits) - 1
        if shared_exp - max_sub_exp > shift_upper_bound:
            shift = shift_upper_bound
        else:
            shift = shared_exp - max_sub_exp

        for j in range(sub_block_size):
            idx = offset + i * sub_block_size + j
            input_x = __float_as_uint(input[idx])
            exp = (input_x & 0x7f800000) >> m_float
            if exp == 0:
                mantissa = 0
            else:
                mantissa = (input_x & 0x7fffff) | (1 << m_float)

            num_bits_shifting = shared_exp - shift - exp + m_float - m_bfp
            if num_bits_shifting >= 32:
                num_bits_shifting = 31
            mantissa >>= num_bits_shifting
            if rounding_mode == 0 and mantissa != ((1 << (m_bfp + 1)) - 1):
                mantissa += 1
            mantissa >>= 1
            sign = -1 if input_x & 0x80000000 else 1
            if shared_exp == 0xff:
                output[idx] = __uint_as_float(0x7fffffff)
            else:
                output[idx] = sign * (2.0 ** int(shared_exp - exp_bias - shift + 1 - m_bfp)) * int(mantissa)