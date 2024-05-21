from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.curdir)
library_name = "kernel_tools"
extensions_dir = os.path.join(this_dir, library_name, "csrc")

USE_JIT = os.getenv('USE_KERNEL_TOOLS_JIT', '0') == '1'

if not USE_JIT:
    setup(
        name='kernel_tools',
        version='0.1',
        description='PyTorch extension with C++ and CUDA',
        packages=['kernel_tools'],
        ext_modules=[
            CUDAExtension(
                name= "kernel_tools_cpp_cuda",
                sources=[
                    'kernel_tools/csrc/tensor_test.cpp', 
                    'kernel_tools/csrc/bindings.cpp',
                ],
                include_dirs=['kernel_tools/csrc'],
                extra_compile_args={'cxx': ['-g'], 'nvcc': ['-g']}
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
