import math
import numpy as np
def cpu_set(N, data, val):
    for index in range(N):
        data[index] = val
    return data

def cpu_scale_inplace(N, data, scale):
    for index in range(N):
        data[index] *= scale
    return data

def cpu_scale(N, src, dst, scale):
    for index in range(N):
        dst[index] = scale * src[index]
    return dst

def cpu_pow(N, data, power):
    for index in range(N):
        data[index] = np.power(data[index], power)
    return data

def cpu_max(N, src):
    dst = src[0]
    for i in range(1, N):
        tmp = src[i]
        if dst < tmp:
            dst = tmp
    return dst

def cpu_min(N, src):
    dst = src[0]
    for i in range(1, N):
        tmp = src[i]
        if dst > tmp:
            dst = tmp
    return dst

def cpu_sum(N, src):
    dst = 0
    for index in range(N):
        dst = dst + src[index]
    return dst

def cpu_sub(N, src, dst):
    for index in range(N):
        dst[index] = src[index] - dst[index]
    return dst
