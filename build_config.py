import os

# Common paths and names for sharing between COMPILED and JIT build styles
NAME="kernel_tools_cpp_cuda"
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'kernel_tools/csrc'))
INCLUDE_DIRS = [SRC_DIR]
SOURCES = [
    os.path.join(SRC_DIR, 'bindings.cpp'),
    os.path.join(SRC_DIR, 'cusolverDnXsyevdx.cu'),
    os.path.join(SRC_DIR, 'cusolverMgSyevd.cu'),
    os.path.join(SRC_DIR, 'utils.cu'),
]

USE_DEBUG = os.getenv('USE_KERNEL_TOOLS_DEBUG', '0') == '1'
USE_VERBOSE = os.getenv('USE_KERNEL_TOOLS_VERBOSE', '0') == '1'

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

VERBOSE = USE_VERBOSE

LIBRARY_DIRS = []

LIBRARIES = [
    'cusolverMg',
    'cusolver',
]

EXTRA_LINK_ARGS = [
    '-L' + path for path in LIBRARY_DIRS
] + [
    '-l' + lib for lib in LIBRARIES
]