"""Golden logit generation and PyTorch CPU shim for DeepSeek-V4.1-Flash."""

from deepseek_v41.golden.torch_cpu_shim import (
    act_quant,
    build_torch_reference_model,
     config_to_model_args,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    install_torch_cpu_kernel_shim,
    load_reference_model_module,
    make_random_reference_weights,
    set_torch_model_modes,
    sparse_attn,
)

__all__ = [
    "act_quant",
    "build_torch_reference_model",
    "config_to_model_args",
    "fp4_act_quant",
    "fp4_gemm",
    "fp8_gemm",
    "hc_split_sinkhorn",
    "install_torch_cpu_kernel_shim",
    "load_reference_model_module",
    "make_random_reference_weights",
    "set_torch_model_modes",
    "sparse_attn",
]
