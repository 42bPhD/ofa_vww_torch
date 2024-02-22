# plase convert 
import math


def vai_round_cpu(x, method):
    if method == 2:  # half_up
        if x < 0 and (x - math.floor(x)) == 0.5:
            return math.ceil(x)
        else:
            return round(x)
    elif method == 3:  # c++ std::round: negative half_down, positive half_up
        return round(x)
    elif method == 4:  # floor
        return math.floor(x)
    elif method == 5:  # negative half_up, positive half_even 
        if x < 0 and (x - math.floor(x)) == 0.5:
            return math.ceil(x)
        elif x - math.floor(x) == 0.5:
            if int(math.floor(x)) % 2 == 0:
                return math.floor(x)
            else:
                return math.ceil(x)
        else:
            return round(x)
    elif method == 6:  # towards zero: negative half_up, positive half_down (vs method 3)
        if x < 0 and (x - math.floor(x)) == 0.5:
            return math.ceil(x)
        elif x > 0 and (x - math.floor(x)) == 0.5:
            return math.floor(x)
        else:
            return round(x)
    elif method == 7:  # up
        return math.ceil(x)
    elif method == 8:  # half_even
        if x < 0 and (x - math.floor(x)) == 0.5:
            if int(math.ceil(x)) % 2 == 0:
                return math.ceil(x)
            else:
                return math.floor(x)
        elif x - math.floor(x) == 0.5:
            if int(math.floor(x)) % 2 == 0:
                return math.floor(x)
            else:
                return math.ceil(x)
        else:
            return round(x)
def mapping_sigm_cpu(output_amp, map_data, src):
    if src >= 8 * output_amp:
        return 32767
    elif src < -8 * output_amp:
        return 0
    else:
        if src >= 0:
            pos = int((src * 1024) // (8 * output_amp))
            pos %= 1024
            if src == -8 * output_amp and pos == 0:
                dst = 0
            else:
                dst = map_data[1023 - pos]
        else:
            pos = int((src * 1024) // (8 * output_amp))
            pos %= 1024
            if src == -8 * output_amp and pos == 0:
                dst = 0
            else:
                dst = map_data[1023 - pos]
        return dst
def fix_neuron_v2_cpu(src, val_min, val_max, val_amp, zero_point, method):
    """
    Method: 
     2: half_up 
     3: c++ std::round: negative half_down, positive half_up
     4: floor
     5: negative half_up, positive half_even
     6: towards zero: negative half_up, positive half_down (vs method 3)
     7: up
     8: half_even 
    """
    res_real_ = src * val_amp
    res = vai_round_cpu(res_real_, method)
    res = res + zero_point
    if res > val_max:
        res = val_max
    elif res < val_min:
        res = val_min
    return res
def mapping_sigm_cpu(output_amp, map_data, src):
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
            if src == -8 * output_amp and pos == 0:
                dst = 0
            else:
                dst = map_data[1024 - pos]
    return dst

def _mappingI_sigm_cpu(output_fp, map_data, src):
    if (src >> output_fp) >= 8:
        return 32767
    elif (src >> output_fp) < -8:
        return 0
    else:
        pos = 0
        if output_fp >= 7:
            pos = src >> (output_fp - 7)
        else:
            pos = src << (7 - output_fp)
        pos %= 2048
        if pos < 0:
            return map_data[2048 + pos]
        else:
            return map_data[pos]
def _mappingI_tanh_cpu(output_fp, map_data, src):
    if (src >> output_fp) >= 4:
        return 32767
    elif (src >> output_fp) < -4:
        return -32768
    else:
        pos = 0
        if output_fp >= 8:
            pos = src >> (output_fp - 8)
        else:
            pos = src << (8 - output_fp)
        pos %= 2048
        if pos < 0:
            return map_data[2048 + pos]
        else:
            return map_data[pos]
def _scaleI_cpu(result, bitwidth, shift):
    if shift > 0:
        result <<= shift
    else:
        result >>= -shift
    max_val = 1 << bitwidth
    if result > max_val - 1:
        return result % max_val - max_val
    elif result < -max_val:
        return max_val + result % (-max_val)
    else:
        return result
def _dimi_floor_cpu(result, val_amp, val_min, val_max):
    result_ = math.floor(result / val_amp)
    if result_ > val_max:
        result_ = val_max
    elif result_ < val_min:
        result_ = val_min
    return result_
def _amp_floor_cpu(result, val_amp, val_min, val_max):
    result_ = math.floor(result * val_amp)
    if result_ > val_max:
        result_ = val_max
    elif result_ < val_min:
        result_ = val_min
    return result_
def _dimi_cpu(result, val_amp):
    return result / val_amp
def _amp_cpu(result, val_amp):
    return result * val_amp
def _floor_cpu(result, val_min, val_max):
    result_ = math.floor(result)
    if result_ > val_max:
        result_ = val_max
    elif result_ < val_min:
        result_ = val_min
    return result_
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
        if method == 1:
            result_ = math.floor(result * val_amp) if not dimi else math.floor(result * (1 / val_amp))
        else:
            result_ = math.ceil(result * val_amp) if not dimi else math.ceil(result * (1 / val_amp))
        if result_ > val_max:
            result_ = val_max
        elif result_ < val_min:
            result_ = val_min
        if keep_scale:
            result = result_ * (1 / val_amp) if not dimi else result_ * val_amp
        else:
            result = result_
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
        if keep_scale:
            result = fixed_result_ * (1 / val_amp) if not dimi else fixed_result_ * val_amp
        else:
            result = fixed_result_
    return result

