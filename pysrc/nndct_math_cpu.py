from pysrc.nndct_cpu_math import cpu_scale_inplace
from pysrc.nndct_fix_kernels import cpu_aie_sqrt, cpu_aie_isqrt, cpu_layernorm_isqrt, cpu_sigmoid_table_lookup, cpu_tanh_table_lookup
from pysrc.nndct_fix_kernels import _tanh_table_lookup, _sigmoid_table_lookup
import torch

def _Scale(Tinput, scale, device_id):
    input = Tinput.data
    num_ele = Tinput.numel()
    return cpu_scale_inplace(num_ele, input, scale)

def Scale(Tinput, scale, device_id):
    if Tinput.dtype == torch.float32:
        return _Scale(Tinput, scale, device_id)
    elif Tinput.dtype == torch.float64:
        return _Scale(Tinput, scale, device_id)

def _SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id):
    input = Tinput.data
    table = Ttable.data
    output = Toutput.data
    num_ele = Tinput.numel()
    return cpu_sigmoid_table_lookup(num_ele, input, table, output, fragpos)

def SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id):
    if Tinput.dtype == torch.float32:
        return _SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
    elif Tinput.dtype == torch.float64:
        return _SigmoidTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)

def _TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id):
    input = Tinput.data
    table = Ttable.data
    output = Toutput.data
    num_ele = Tinput.numel()
    return cpu_tanh_table_lookup(num_ele, input, table, output, fragpos)

def TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id):
    if Tinput.dtype == torch.float32:
        return _TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)
    elif Tinput.dtype == torch.float64:
        return _TanhTableLookup(Tinput, Ttable, Toutput, fragpos, device_id)


def _SigmoidSimulation(Tinput, Toutput, fragpos, device_id):
    print("Sigmoid simulation is not supported in CPU mode.")

def SigmoidSimulation(Tinput, Toutput, fragpos, device_id):
    if Tinput.dtype == torch.float32:
        _SigmoidSimulation(Tinput, Toutput, fragpos, device_id)
    elif Tinput.dtype == torch.float64:
        _SigmoidSimulation(Tinput, Toutput, fragpos, device_id)

def _TanhSimulation(Tinput, Toutput, fragpos, device_id):
    print("Tanh simulation is not supported in CPU mode.")

def TanhSimulation(Tinput, Toutput, fragpos, device_id):
    if Tinput.dtype == torch.float32:
        _TanhSimulation(Tinput, Toutput, fragpos, device_id)
    elif Tinput.dtype == torch.float64:
        _TanhSimulation(Tinput, Toutput, fragpos, device_id)

def _SoftmaxExpApproximate(Tinput, Toutput, device_id):
    print("Softmax Exponent Approximate is not supported in CPU mode.")

def SoftmaxExpApproximate(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _SoftmaxExpApproximate(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _SoftmaxExpApproximate(Tinput, Toutput, device_id)

def _SoftmaxLOD(Tinput, Toutput, device_id):
    print("Softmax LOD is not supported in CPU mode.")

def SoftmaxLOD(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _SoftmaxLOD(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _SoftmaxLOD(Tinput, Toutput, device_id)

def _SoftmaxSimulationPart1(Tinput, Toutput, device_id):
    print("Softmax Simulation Part 1 is not supported in CPU mode.")

def SoftmaxSimulationPart1(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _SoftmaxSimulationPart1(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _SoftmaxSimulationPart1(Tinput, Toutput, device_id)

def _SoftmaxSimulationPart2(sum, Toutput, device_id):
    print("Softmax Simulation Part 2 is not supported in CPU mode.")

def SoftmaxSimulationPart2(sum, Toutput, device_id):
    if Toutput.dtype == torch.float32:
        _SoftmaxSimulationPart2(sum, Toutput, device_id)
    elif Toutput.dtype == torch.float64:
        _SoftmaxSimulationPart2(sum, Toutput, device_id)

def _SigmoidTableLookupAIE2(Tinput, Toutput, fragpos, device_id):
    print("Sigmoid Table Look up AIE2 is not supported in CPU mode.")

def SigmoidTableLookupAIE2(Tinput, Toutput, fragpos, device_id):
    if Tinput.dtype == torch.float32:
        _SigmoidTableLookupAIE2(Tinput, Toutput, fragpos, device_id)
    elif Tinput.dtype == torch.float64:
        _SigmoidTableLookupAIE2(Tinput, Toutput, fragpos, device_id)

def _TanhTableLookupAIE2(Tinput, Toutput, fragpos, device_id):
    print("Tanh Table Look up AIE2 is not supported in CPU mode.")

def TanhTableLookupAIE2(Tinput, Toutput, fragpos, device_id):
    if Tinput.dtype == torch.float32:
        _TanhTableLookupAIE2(Tinput, Toutput, fragpos, device_id)
    elif Tinput.dtype == torch.float64:
        _TanhTableLookupAIE2(Tinput, Toutput, fragpos, device_id)

def _ExpApprAIE2(Tinput, Toutput, bit_width, device_id):
    print("Exp Approximation AIE2 is not supported in CPU mode.")

def ExpApprAIE2(Tinput, Toutput, bit_width, device_id):
    if Tinput.dtype == torch.float32:
        _ExpApprAIE2(Tinput, Toutput, bit_width, device_id)
    elif Tinput.dtype == torch.float64:
        _ExpApprAIE2(Tinput, Toutput, bit_width, device_id)

def _LogSoftmaxFastLn(Tinput, Toutput, device_id):
    print("LogSoftmax Ln is not supported in CPU mode.")

def LogSoftmaxFastLn(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _LogSoftmaxFastLn(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _LogSoftmaxFastLn(Tinput, Toutput, device_id)

def _LogSoftmaxSub(Tinput, Toutput, Tsum, device_id):
    print("LogSoftmax Ln is not supported in CPU mode.")

def LogSoftmaxSub(Tinput, Toutput, Tsum, device_id):
    if Tinput.dtype == torch.float32:
        _LogSoftmaxSub(Tinput, Toutput, Tsum, device_id)
    elif Tinput.dtype == torch.float64:
        _LogSoftmaxSub(Tinput, Toutput, Tsum, device_id)

def _AIESqrt(Tinput, Toutput, device_id):
    print("AIE Sqrt is not supported in CPU mode.")

def AIESqrt(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _AIESqrt(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _AIESqrt(Tinput, Toutput, device_id)

def _AIEISqrt(Tinput, Toutput, device_id):
    print("AIE ISqrt is not supported in CPU mode.")

def AIEISqrt(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _AIEISqrt(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _AIEISqrt(Tinput, Toutput, device_id)

def _LayernormISqrt(Tinput, Toutput, device_id):
    print("Layernorm ISqrt is not supported in CPU mode.")

def LayernormISqrt(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _LayernormISqrt(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _LayernormISqrt(Tinput, Toutput, device_id)

def _LayernormInvSqrt(Tinput, Toutput, device_id):
    print("Layernorm InvSqrt is not supported in CPU mode.")

def LayernormInvSqrt(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _LayernormInvSqrt(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _LayernormInvSqrt(Tinput, Toutput, device_id)

def _InverseAIE2(Tinput, Toutput, device_id):
    print("Inverse AIE2 is not supported in CPU mode.")

def InverseAIE2(Tinput, Toutput, device_id):
    if Tinput.dtype == torch.float32:
        _InverseAIE2(Tinput, Toutput, device_id)
    elif Tinput.dtype == torch.float64:
        _InverseAIE2(Tinput, Toutput, device_id)