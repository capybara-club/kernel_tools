from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from build_config import NAME, INCLUDE_DIRS, SOURCES, EXTRA_COMPILE_ARGS, EXTRA_LINK_ARGS

this_dir = os.path.dirname(os.path.curdir)
library_name = "kernel_tools"
extensions_dir = os.path.join(this_dir, library_name, "csrc")

USE_JIT = os.getenv('USE_KERNEL_TOOLS_JIT', '0') == '1'

def get_ext_modules():
    if USE_JIT:
        return []
    else:
        return [
            CUDAExtension(
                name= NAME,
                sources=SOURCES,
                include_dirs=INCLUDE_DIRS,
                extra_compile_args=EXTRA_COMPILE_ARGS,
                extra_link_args=EXTRA_LINK_ARGS,
            ),
        ]

setup(
    name='kernel_tools',
    version='0.1',
    description='PyTorch extension with C++ and CUDA',
    packages=['kernel_tools'],
    ext_modules=get_ext_modules(),
    cmdclass={
        'build_ext': BuildExtension
    }
)