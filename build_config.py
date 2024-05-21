import os

# Common paths and names for sharing between COMPILED and JIT build styles
NAME="kernel_tools_cpp_cuda"
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'kernel_tools/csrc'))
INCLUDE_DIRS = [SRC_DIR]
SOURCES = [
    os.path.join(SRC_DIR, 'bindings.cpp'),
    os.path.join(SRC_DIR, 'tensor_test.cpp')
]

USE_DEBUG = os.getenv('USE_KERNEL_TOOLS_DEBUG', '0') == '1'

def get_cxx_compile_args():
    if USE_DEBUG:
        return ['-g']
    return [
        '-O3',
    ]

def get_nvcc_compile_args():
    if USE_DEBUG:
        return ['-g']
    return [
        '-O3',
    ]

EXTRA_COMPILE_ARGS = {
    'cxx': get_cxx_compile_args(),
    'nvcc': get_nvcc_compile_args()
}