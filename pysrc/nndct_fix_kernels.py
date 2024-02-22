
import math
import numpy as np
def cpu_vai_round(N, src, method):
    return _cpu_vai_round(N, src, method)

def _cpu_vai_round(N, src, method):
    dst = [_vai_round_cpu(src[index], method) for index in range(N)]
    # for index in range(N):
    #     result_ = _vai_round_cpu(src[index], method)
    #     dst.append(result_)
    return dst


def cpu_fix_neuron_v1(N, src, fragpos, dst, val_min, val_max, keep_scale, method):
    return _fix_neuron_v1(N, src, fragpos, dst, val_min, val_max, keep_scale, method)

def cpu_fix_neuron_v2(N, src, dst, val_min, val_max, val_amp, zero_point, keep_scale, method):
    return _fix_neuron_v2(N, src, dst, val_min, val_max, val_amp, zero_point, keep_scale, method)
    
def _fix_neuron_v1(N, src, fragpos, dst, val_min, val_max, keep_scale, method):
    for index in range(N):
        result = 0
        val_amp = np.power(2, fragpos)
        result = _fix_neuron_v2_cpu(src[index], result, val_min, val_max, val_amp, 0, method)
        if keep_scale != 0:
            dst[index] = result * (1 / val_amp)
        else:
            dst[index] = result
    return dst
# def _fix_neuron_v2(N:int, src:torch.Tensor, dst:torch.Tensor, 
#                    val_min:int, val_max:int, val_amp:int, zero_point:int, keep_scale:int, method:int):
#     tensor_size = src.size()
#     src = torch.flatten(src)
#     dst = torch.zeros_like(src)
#     result = _fix_neuron_v2_cpu(src, 
#                                 torch.zeros_like(src), 
#                                 val_min, 
#                                 val_max, 
#                                 val_amp, 
#                                 zero_point, 
#                                 method)
#     if keep_scale != 0:
#         dst = (result - zero_point) * (1 / val_amp)
#     else:
#         dst = result
#     dst = torch.reshape(dst, tensor_size)
#     return dst
def _fix_neuron_v2(N:int, src, dst, 
                   val_min:int, val_max:int, val_amp:int, zero_point:int, keep_scale:int, method:int):
    result = _fix_neuron_v2_cpu(src, 
                                torch.zeros_like(src), 
                                val_min, 
                                val_max, 
                                val_amp, 
                                zero_point, 
                                method)
    
    if keep_scale != 0:
        dst = (result - zero_point) * (1 / val_amp)
    else:
        dst = result
    return dst

def cpu_diff_S(N, src, buffer, output, bitwidth, range, method):
    # Calc search range for scale
    fix_lb = -np.power(2, bitwidth - 1) - 0.5
    fix_ub = np.power(2, bitwidth - 1) - 0.5

    x_max = np.max(src)
    x_min = np.min(src)

    # Find max_scale
    step = max(x_min / fix_lb, x_max / fix_ub)
    if step <= np.finfo(float).eps:
        max_scale = 18
    else:
        max_scale = np.floor(np.log2(1 / step))

    # Find fix pos in range [max_scale + range , max_scale]
    final_scale = max_scale
    fixed_diff_min = np.finfo(float).max
    for scale in range(max_scale, max_scale + range):
        buffer = cpu_fix_neuron_v2(N, src, buffer, -(1<<(bitwidth-1)), (1<<(bitwidth-1))-1, np.power(2, scale), 0, 1, method)
        buffer = np.subtract(src, buffer)
        buffer = np.power(buffer, 2)
        fixed_diff = np.sum(buffer)
        if fixed_diff < fixed_diff_min:
            final_scale = scale
            fixed_diff_min = fixed_diff

    output[0] = final_scale
    return output


### nndct_fix_kernels_cpu.h

import numpy as np

def _sigmoid_table_lookup(N, fragpos, scale, fuzz, input, table):
    output = np.zeros_like(input)
    for i in range(N):
        if input[i] >= 8.0:
            output[i] = 1.0 - fuzz
        elif input[i] < -8.0:
            output[i] = 0.0
        else:
            x = int(input[i] * scale)
            pos = 0
            if x >= 0:
                if fragpos >= 7:
                    pos = (x >> (fragpos - 7)) % 1024
                else:
                    pos = (x << (7 - fragpos)) % 1024
                output[i] = table[pos + 1024] * fuzz
            else:
                # if (fragpos >= 7):
                #     pos = (abs(x) >> (fragpos - 7)) % 1024
                # else:
                #     pos = (x << (7 - fragpos)) % 1024
                
                pos = abs(int(np.floor(x / (2.0 ** (fragpos - 7))))) % 1024
                if (x >> fragpos) == -8 and pos == 0:
                    output[i] = table[pos] * fuzz
                else:
                    output[i] = table[1024 - pos] * fuzz
    return output

def cpu_sigmoid_table_lookup(N, input, table, fragpos):
    scale = 2.0 ** fragpos
    fuzz = 1.0 / 32768
    return _sigmoid_table_lookup(N, fragpos, scale, fuzz, input, table)

def _tanh_table_lookup(N, fragpos, scale, fuzz, input, table, output):
    output = np.zeros_like(input)
    for i in range(N):
        if input[i] >= 4.0:
            output[i] = 1.0 - fuzz
        elif input[i] < -4.0:
            output[i] = -1.0
        else:
            x = int(input[i] * scale)
            pos = 0
            if x >= 0:
                if fragpos >= 8:
                    pos = (x >> (fragpos - 8)) % 1024
                else:
                    pos = (x << (8 - fragpos)) % 1024
                output[i] = table[pos + 1024] * fuzz
            else:
                pos = abs(int(math.floor(x / math.pow(2.0, (fragpos - 8))))) % 1024
                if (x >> fragpos) == -4 and pos == 0:
                    output[i] = table[pos] * fuzz
                else:
                    output[i] = table[1024 - pos] * fuzz
    return output

def cpu_tanh_table_lookup(N, input, table, output, fragpos):
    scale = torch.pow(2.0, fragpos)
    fuzz = 1.0 / 32768
    return _tanh_table_lookup(N, fragpos, scale, fuzz, input, table, output)


def _mapping_sigm_cpu(output_amp, map_data, src):
    if src >= 8 * output_amp:
        dst = 32767
    elif src < -8 * output_amp:
        dst = 0
    else:
        if src >= 0:
            pos = 0
            if output_amp >= 128:
                pos = math.floor(src / (output_amp / 128))
            else:
                pos = math.floor(src * (128 / output_amp))
            pos %= 1024
            dst = map_data[1024 + pos]
        else:
            pos = 0
            if output_amp >= 128:
                pos = math.floor(abs(src) / (output_amp / 128))
            else:
                pos = math.floor(abs(src) * (128 / output_amp))
            pos %= 1024
            if (src == -8 * output_amp) and pos == 0:
                dst = 0
            else:
                dst = map_data[1024 - pos]
    return dst

def _mapping_tanh_cpu(output_amp, map_data, src):
    if src >= 4 * output_amp:
        dst = 32767
    elif src < -4 * output_amp:
        dst = -32768
    else:
        if src >= 0:
            pos = 0
            if output_amp >= 256:
                pos = math.floor(src / (output_amp / 256))
            else:
                pos = math.floor(src * (256 / output_amp))
            pos %= 1024
            dst = map_data[1024 + pos]
        else:
            pos = 0
            if output_amp >= 256:
                pos = math.floor(abs(src) / (output_amp / 256))
            else:
                pos = math.floor(abs(src) * (256 / output_amp))
            pos %= 1024
            if (src == -4 * output_amp) and (pos == 0):
                dst = map_data[pos]
            else:
                dst = map_data[1024 - pos]
    return dst

def _mappingI_sigm_cpu(output_fp, map_data, src):
    if (src >> output_fp) >= 8:
        dst = 32767
    elif (src >> output_fp) < -8:
        dst = 0
    else:
        pos = 0
        if output_fp >= 7:
            pos = src >> (output_fp - 7)
        else:
            pos = src << (7 - output_fp)
        pos %= 2048
        if pos < 0:
            dst = map_data[2048 + pos]
        else:
            dst = map_data[pos]
    return dst

def _mappingI_tanh_cpu(output_fp, map_data, src):
    if (src >> output_fp) >= 4:
        dst = 32767
    elif (src >> output_fp) < -4:
        dst = -32768
    else:
        pos = 0
        if output_fp >= 8:
            pos = src >> (output_fp - 8)
        else:
            pos = src << (8 - output_fp)
        pos %= 2048
        if pos < 0:
            dst = map_data[2048 + pos]
        else:
            dst = map_data[pos]
    return dst
def _scaleI_cpu(result, bitwidth, shift):
    if shift > 0:
        result <<= shift
    else:
        result >>= -shift
    max_val = 1 << bitwidth
    if result > max_val - 1:
        result = result % max_val - max_val
    elif result < -max_val:
        result = max_val + result % -max_val
    return result

def _dimi_floor_cpu(result, val_amp, val_min, val_max):
    result_ = math.floor(result / val_amp)
    if result_ > val_max:
        result_ = val_max
    elif result_ < val_min:
        result_ = val_min
    return float(result_)

def _amp_floor_cpu(result, val_amp, val_min, val_max):
    result_ = math.floor(result * val_amp)
    if result_ > val_max:
        result_ = val_max
    elif result_ < val_min:
        result_ = val_min
    return float(result_)

def _dimi_cpu(result, val_amp):
    result /= val_amp
    return result

def _amp_cpu(result, val_amp):
    result *= val_amp
    return result

def _floor_cpu(result, val_min, val_max):
    result_ = math.floor(result)
    if result_ > val_max:
        result_ = val_max
    elif result_ < val_min:
        result_ = val_min
    return float(result_)

def _dimiI_cpu(result, diff_amp):
    tmp_result = int(result / diff_amp) if diff_amp >= 1 else int(result * diff_amp)
    if diff_amp > 1 and result % int(diff_amp) != 0 and result < 0:
        tmp_result -= 1
    return tmp_result

def _dimiI_floor_cpu(result, val_amp, val_min, val_max):
    result /= val_amp
    if result > val_max:
        result = val_max
    elif result < val_min:
        result = val_min
    return result

def _fix_neuron_v2_cpu_tmp(result, val_amp, val_min, val_max, dimi, keep_scale, method):
    if method == 0:
        result = result * val_amp if not dimi else result * (1 / val_amp)
    elif method == 1 or method == 3:
        result_ = math.floor(result * val_amp) if not dimi else math.floor(result * (1 / val_amp))
        if result_ > val_max:
            result_ = val_max
        elif result_ < val_min:
            result_ = val_min
        result = float(result_) * (1 / val_amp) if not dimi else float(result_) * val_amp if keep_scale else result_
    elif method == 2:
        result_ = result * val_amp if not dimi else result * (1 / val_amp)
        if result_ > val_max:
            result_ = val_max
        elif result_ < val_min:
            result_ = val_min
        elif result_ < 0 and (result_ - math.floor(result_)) == 0.5:
            fixed_result_ = math.ceil(result_)
        else:
            fixed_result_ = round(result_)
        result = float(fixed_result_) * (1 / val_amp) if not dimi else float(fixed_result_) * val_amp if keep_scale else fixed_result_
    return result


import struct
import numpy as np

def float2int_cpu(x):
    return struct.unpack('i', struct.pack('f', x))[0]

def int2float_cpu(x):
    return struct.unpack('f', struct.pack('i', x))[0]

def int2bfloat_cpu(x):
    itmp = x
    if (itmp & 0x00008000) == 0x00008000:
        if (itmp & 0xFFFF) > 0x00008000 or (((itmp & 0xFFFF) == 0x00008000) and (itmp & 0x10000) == 0x10000):
            itmp += 0x10000
    itmp &= 0xFFFF0000
    return int2float_cpu(itmp)

def float2bfloat_cpu(x):
    itmp = float2int_cpu(x)
    if (itmp & 0x00008000) == 0x00008000:
        if (itmp & 0xFFFF) > 0x00008000 or (((itmp & 0xFFFF) == 0x00008000) and (itmp & 0x10000) == 0x10000):
            itmp += 0x10000
    itmp &= 0xFFFF0000
    return int2float_cpu(itmp)

def float2short_cpu(x):
    itmp = float2bfloat_cpu(x)
    return float2int_cpu(itmp)

def short_downshift_onebit_cpu(x):
    y = x >> 17
    a = (x >> 16) & 1
    if (y & 1) == 1:
        y += a
    return y << 17

def _sqrt(x):
    x2 = x * 0.5
    y = x
    i = float2int_cpu(y)
    i = (0x5f37 - (i >> 17)) << 16
    y = int2float_cpu(i)
    y3h = float2bfloat_cpu(1.5 * y)
    out = float2bfloat_cpu(y * x2)
    out = float2bfloat_cpu(out * y)
    out = float2bfloat_cpu(out * y)
    out = float2bfloat_cpu(y3h - out)
    out = float2bfloat_cpu(x * out)
    return out

def _aie_sqrt(N, src, dst):
    for index in range(N):
        result_ = 1.0
        dst[index] = _sqrt(src[index])

def cpu_aie_sqrt(N, src, dst):
    _aie_sqrt(N, src, dst)

def _isqrt(x):
    x2 = float2bfloat_cpu(x * 0.5)
    i = float2short_cpu(x)
    i = (0x5f37 - (short_downshift_onebit_cpu(i) >> 17)) << 16
    y = int2bfloat_cpu(i)
    threehalfs = float2bfloat_cpu(1.5)
    for _ in range(4):
        y2 = float2bfloat_cpu(y * y)
        mul2 = float2bfloat_cpu(x2 * y2)
        sub = float2bfloat_cpu(threehalfs - mul2)
        y = float2bfloat_cpu(y * sub)
    return y

def _aie_isqrt(N, src, dst):
    for index in range(N):
        dst[index] = _isqrt(src[index])

def cpu_aie_isqrt(N, src, dst):
    _aie_isqrt(N, src, dst)

def isqrt(x):
    x2 = x * 0.5
    y = x
    threehalfs = 1.5
    i = float2int_cpu(y)
    i = 0x5f3759df - (i >> 1)
    y = int2float_cpu(i)
    for _ in range(4):
        y = y * (threehalfs - (x2 * y * y))
    return y

def _layernorm_isqrt(N, src, dst):
    for index in range(N):
        dst[index] = isqrt(src[index])

def cpu_layernorm_isqrt(N, src, dst):
    _layernorm_isqrt(N, src, dst)
    
    
import torch
def _vai_round_cpu(x, method):
    if method == 2:  # half_up
        return torch.where((x < 0) & ((x - torch.floor(x)) == 0.5), torch.ceil(x), torch.round(x))
    elif method == 3:  # c++ std::round: negative half_down, positive half_up
        return torch.round(x)
    elif method == 4:  # floor
        return torch.floor(x)
    elif method == 5:  # negative half_up, positive half_even
        return torch.where((x < 0) & ((x - torch.floor(x)) == 0.5), torch.ceil(x), torch.where((x - torch.floor(x)) == 0.5, torch.where(torch.floor(x) % 2 == 0, torch.floor(x), torch.ceil(x)), torch.round(x)))
    elif method == 6:  # towards zero: negative half_up, positive half_down (vs method 3)
        return torch.where((x < 0) & ((x - torch.floor(x)) == 0.5), torch.ceil(x), torch.where((x > 0) & ((x - torch.floor(x)) == 0.5), torch.floor(x), torch.round(x)))
    elif method == 7:  # up
        return torch.ceil(x)
    elif method == 8:  # half_even
        return torch.where((x < 0) & ((x - torch.floor(x)) == 0.5), torch.where(torch.ceil(x) % 2 == 0, torch.ceil(x), torch.floor(x)), torch.where((x - torch.floor(x)) == 0.5, torch.where(torch.floor(x) % 2 == 0, torch.floor(x), torch.ceil(x)), torch.round(x)))
    

def _fix_neuron_v2_cpu(src, res, val_min, val_max, val_amp, zero_point, method):
    res_real_ = src * val_amp
    res = _vai_round_cpu(res_real_, method)
    res = res + zero_point
    return torch.where(res > val_max, val_max, torch.where(res < val_min, val_min, res))
    # if res > val_max:
    #     res = val_max
    # elif res < val_min:
    #     res = val_min
    # return res
