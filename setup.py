from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import os
from build_config import NAME, INCLUDE_DIRS, SOURCES, EXTRA_COMPILE_ARGS, EXTRA_LINK_ARGS


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
    packages=find_packages(),
    description='PyTorch extension with C++ and CUDA',
    ext_modules=get_ext_modules(),
    cmdclass={
        'build_ext': BuildExtension
    },
    # package_data={
    #     'kernel_tools.csrc': ['*.cpp', '*.h', '*.cu'],
    # },
    # include_package_data=True,
)