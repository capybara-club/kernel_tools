import torch
from torch.utils.cpp_extension import load, load_inline
import os
import sys

# JIT Compilation
def load_jit():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from build_config import NAME, INCLUDE_DIRS, SOURCES, EXTRA_COMPILE_ARGS

    kernel = load(
        name=NAME,
        sources=SOURCES,
        extra_include_paths=INCLUDE_DIRS,
        extra_cflags=EXTRA_COMPILE_ARGS['cxx'],
        extra_cuda_cflags=EXTRA_COMPILE_ARGS['nvcc'],
        verbose=True
    )
    return kernel

USE_JIT = os.getenv('USE_KERNEL_TOOLS_JIT', '0') == '1'

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
