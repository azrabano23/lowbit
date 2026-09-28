"""lowbit: fused dequantize + GEMM CPU kernels for low-bit LLM weight formats."""
from .formats import (GROUP, MXBLOCK, Int4Weights, Int8Acts, MXFP4Weights, quantize_act_int8,
                      quantize_int4, quantize_mxfp4)
from .kernels import Params, cpu_info, gemm_mxfp4, gemm_w4a8, gemm_w4a16

__version__ = "0.1.0"
__all__ = [
    "GROUP", "MXBLOCK", "Int4Weights", "Int8Acts", "MXFP4Weights", "Params", "cpu_info",
    "gemm_mxfp4", "gemm_w4a8", "gemm_w4a16", "quantize_act_int8", "quantize_int4",
    "quantize_mxfp4",
]
