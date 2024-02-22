import torch
from pysrc.nndct_fix_kernels import cpu_diff_S

def _DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id):
    input = Tinput.data
    buffer = Tbuffer.data
    fixpos = Tfixpos.data
    num_ele = Tinput.numel()
      
    cpu_diff_S(num_ele, input, buffer, fixpos, bit_width, range, method)

def DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id):
    if Tinput.dtype == torch.float32:
        _DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)
    elif Tinput.dtype == torch.float64:
        _DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)

def diffs_fix_pos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id):
    if Tinput.dtype == torch.float32:
        _DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)
    elif Tinput.dtype == torch.float64:
        _DiffsFixPos(Tinput, Tbuffer, Tfixpos, bit_width, range, method, device_id)
