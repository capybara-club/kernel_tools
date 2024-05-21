import torch
from torch.utils.cpp_extension import load, load_inline
import os

USE_JIT = os.getenv('USE_KERNEL_TOOLS_JIT', '0') == '1'

# JIT Compilation
def load_jit():
    this_dir = os.path.dirname(os.path.curdir)
    library_name = "kernel_tools"
    extensions_dir = os.path.join(this_dir, library_name, "csrc")

    kernel = load(
        name="kernel_tools_cpp_cuda",
        sources=[
                'kernel_tools/csrc/tensor_test.cpp', 
                'kernel_tools/csrc/bindings.cpp',
        ],
        extra_include_paths=["kernel_tools/csrc"],
        extra_cflags=['-g'],
        extra_cuda_cflags=['-g'],
        verbose=True
    )
    return kernel

# Installation via pip
if USE_JIT:
    print('Going JIT')
    kernel = load_jit()
else:
    try:
        print('Going Compiled')
        import kernel_tools_cpp_cuda as kernel
    except ImportError:
        print('Going JIT')
        kernel = load_jit()

def tensor_test(input):
    return kernel.tensor_test(input)
