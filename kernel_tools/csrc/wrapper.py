from torch.utils.cpp_extension import load, load_inline
import os
import sys
import subprocess

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

# JIT Compilation
def load_jit():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from build_config import NAME, INCLUDE_DIRS, SOURCES, EXTRA_COMPILE_ARGS, VERBOSE

    kernel = load(
        name=NAME,
        sources=SOURCES,
        extra_include_paths=INCLUDE_DIRS,
        extra_cflags=EXTRA_COMPILE_ARGS['cxx'],
        extra_cuda_cflags=EXTRA_COMPILE_ARGS['nvcc'],
        verbose=VERBOSE
    )
    return kernel

USE_JIT = os.getenv('USE_KERNEL_TOOLS_JIT', '0') == '1'

@singleton
class SingletonClass:
    def __init__(self):
        try:
            # Installation via pip
            if USE_JIT:
                print('Running kernel_tools JIT (first run can take a minute)')
                kernel = load_jit()
            else:
                try:
                    print('Running kernel_tools Compiled')
                    import kernel_tools_cpp_cuda as kernel
                except ImportError:
                    print('Running kernel_tools Compiled failed, Going JIT (can take a minute)')
                    kernel = load_jit()
            self.kernel = kernel
        except subprocess.CalledProcessError as e:
            print("Error during compilation:")
            print(e.output.decode())

# If not compiled yet, forces JIT to run now rather than lazily on demand
def warmup():
    SingletonClass()

def tensor_test(input):
    return SingletonClass().kernel.tensor_test(input)
