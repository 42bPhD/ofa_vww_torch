import numpy as np
import struct
import torch
from pysrc.nndct_fix_kernels import *
def _Round(Tinput, Toutput, method):
    input = Tinput.data
    output = Toutput.data
    num_ele = Tinput.numel()

    # cpu_vai_round(num_ele, input, output, method) 
    return cpu_vai_round(num_ele, input, output, method)
    

def Round(Tinput, Toutput, method, device_id):
    return _Round(Tinput, Toutput, method)

def _FixNeuronV2(Tinput, Toutput, valmin, valmax, valamp, zero_point, method, device_id):
    input = Tinput.data
    output = Toutput.data
    num_ele = Tinput.numel()

    return cpu_fix_neuron_v2(num_ele, input, output, valmin, valmax, valamp, zero_point, 1, method)

def FixNeuronV2(Tinput, Toutput, valmin, valmax, valamp, zero_point, method, device_id):
    return _FixNeuronV2(Tinput, Toutput, valmin, valmax, valamp, zero_point, method, device_id)
    

def fix_neuron(Tinput, valmin, valmax, valamp, zero_point, method, device_id, inplace):
    if inplace != 0:
        if Tinput.dtype == torch.float32:
            Tinput = _FixNeuronV2(Tinput, Tinput, valmin, valmax, valamp, zero_point, method, device_id)
        elif Tinput.dtype == torch.float64:
            Tinput = _FixNeuronV2(Tinput, Tinput, valmin, valmax, valamp, zero_point, method, device_id)
        else:
            print("Unsupported tensor type: ", Tinput.dtype)
        return Tinput
    else:
        Toutput = torch.empty_like(Tinput)
        if Tinput.dtype == torch.float32:
            Tinput = _FixNeuronV2(Tinput, Toutput, valmin, valmax, valamp, zero_point, method, device_id)
        elif Tinput.dtype == torch.float64:
            Tinput = _FixNeuronV2(Tinput, Toutput, valmin, valmax, valamp, zero_point, method, device_id)
        else:
            print("Unsupported tensor type: ", Tinput.dtype)
        return Toutput
    
def fix_neuron_per_channel(Tinput, valmin, valmax, scale, zero_point, axis, method, device_id, inplace):
    if Tinput.dtype != torch.float32 and Tinput.dtype != torch.float64:
        raise ValueError("Unsupported tensor type: " + str(Tinput.dtype))

    Tinput_split = torch.split(Tinput, 1, axis)
    if inplace != 0:
        for i in range(len(Tinput_split)):
            scale_i = scale[i].item()
            valamp_i = 1.0 / scale_i
            zero_point_i = int(zero_point[i].item())
            if Tinput.dtype == torch.float32:
                Tinput_split[i] = _FixNeuronV2(Tinput_split[i], Tinput_split[i], valmin, valmax, valamp_i, zero_point_i, method, device_id)
            elif Tinput.dtype == torch.float64:
                Tinput_split[i] =  _FixNeuronV2(Tinput_split[i], Tinput_split[i], valmin, valmax, valamp_i, zero_point_i, method, device_id)
        Toutput = torch.cat(Tinput_split, axis)
        return Toutput
    else:
        Toutput_vector = []
        for i in range(len(Tinput_split)):
            Toutput_i = torch.empty_like(Tinput_split[i])
            scale_i = scale[i].item()
            valamp_i = 1.0 / scale_i
            zero_point_i = int(zero_point[i].item())
            if Tinput.dtype == torch.float32:
                Toutput_i = _FixNeuronV2(Tinput_split[i], Toutput_i, valmin, valmax, valamp_i, zero_point_i, method, device_id)
            elif Tinput.dtype == torch.float64:
                Toutput_i = _FixNeuronV2(Tinput_split[i], Toutput_i, valmin, valmax, valamp_i, zero_point_i, method, device_id)
            Toutput_vector.append(Toutput_i)
        Toutput = torch.cat(Toutput_vector, axis)
        return Toutput
    
def float_as_uint(x):
    return struct.unpack('I', struct.pack('f', x))[0]

def uint_as_float(x):
    return struct.unpack('f', struct.pack('I', x))[0]

def get_exponent_cpu(v):
    uint_v = float_as_uint(v)
    return (uint_v & 0x7f800000) >> 23

def get_max_exponent_cpu(input, n):
    max_exp = 0
    for i in range(n):
        max_exp = max(max_exp, get_exponent_cpu(input[i]))
    return max_exp

def bfp_cpu_kernel(input, output, n, index, stride, bit_width):
    shared_exp = 0
    for i in range(index, n, stride):
        exp = get_exponent_cpu(input[i])
        if exp == 0xff:
            exp = 0
        if exp > shared_exp:
            shared_exp = exp

    shared_exp_value = int(shared_exp) - 127
    m_bits = bit_width - 9
    scale = np.power(2.0, shared_exp_value - (m_bits - 1))
    max_v = np.power(2.0, shared_exp_value + 1) - scale
    for i in range(index, n, stride):
        exp = get_exponent_cpu(input[i])
        if exp == 0xff:
            output[i] = input[i]
        else:
            x = round(input[i] / scale) * scale
            output[i] = max(-max_v, min(x, max_v))

def launch_bfp_cpu_kernel(input, output, n, bit_width, block_size):
    num_blocks = n // block_size
    for index in range(num_blocks):
        bfp_cpu_kernel(input, output, n, index, num_blocks, bit_width)
