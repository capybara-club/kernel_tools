from torch.utils.cpp_extension import load, load_inline
import os
import sys
import subprocess
import torch
import torch.cuda as cuda

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
    from build_config import NAME, INCLUDE_DIRS, SOURCES, EXTRA_COMPILE_ARGS, VERBOSE, EXTRA_LINK_ARGS

    kernel = load(
        name=NAME,
        sources=SOURCES,
        extra_include_paths=INCLUDE_DIRS,
        extra_cflags=EXTRA_COMPILE_ARGS['cxx'],
        extra_cuda_cflags=EXTRA_COMPILE_ARGS['nvcc'],
        extra_ldflags=EXTRA_LINK_ARGS,
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
                except ImportError as e:
                    print(f"ImportError: {e}")
                    print('Running kernel_tools Compiled failed, Going JIT (can take a minute)')
                    kernel = load_jit()
            self.kernel = kernel
        except subprocess.CalledProcessError as e:
            print("Error during compilation:")
            print(e.output.decode())

# If not compiled yet, forces JIT to run now rather than lazily on demand
def warmup():
    SingletonClass()

#TODO: need to handle triangle non-sense with row-major and strange strides
#TODO: handle output tensor and let it be a to avoid allocate
#TODO: specify range from python
#TODO: able to switch to eigenvalue only mode
# On info from docs:
# If output parameter info = -i (less than zero), the i-th parameter is wrong (not counting handle).
# If info = i (greater than zero), i off-diagonal elements of an intermediate tridiagonal 
# form did not converge to zero.

# If out is None, then we'll do a clone of the tensor to not overwrite. 
# If out is a, we don't have to do a clone, these functions are meant to 
# write in place.

def syevdx(a, num_eigs, out=None, is_upper_triangle=True):
    w = torch.zeros(a.size(0), dtype=a.dtype, device=a.device)
    info = torch.scalar_tensor(-1, device=a.device, dtype=torch.int)
    if out is None:
        out = a.clone()
    elif a is not out:
        if (a.shape != out.shape):
            raise ValueError("shape of a and out don't match")
        out.copy_(a)

    stream = cuda.current_stream()

    SingletonClass().kernel.cusolverDnXsyevdx_export(out, w, info, num_eigs, is_upper_triangle, stream.cuda_stream)

    cpu_info = info.cpu()
    if cpu_info != 0:
        raise ValueError(f"info of syevdx is not equal to 0: {cpu_info}. Check cusolver docs")
    return out, w

def mgSyevd(a, d):
    return SingletonClass().kernel.cusolverMgSyevd_export(a, d)