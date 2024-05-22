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

#TODO: need to check triangle non-sense with row-major and strange strides
#TODO: able to switch to eigenvalue only mode
# On info from docs:
# If output parameter info = -i (less than zero), the i-th parameter is wrong (not counting handle).
# If info = i (greater than zero), i off-diagonal elements of an intermediate tridiagonal 
# form did not converge to zero.

# If out is None, then we'll do a clone of the tensor to not overwrite. 
# If out is a, we don't have to do a clone, these functions are meant to 
# write in place.

# indices are 1 indexed

# Trying to match scipy eigh
def syevdx(
        a,
        overwrite_a = False,
        subset_by_index = None, 
        lower = True, 
        eigvals_only = False
):

    if not a.is_cuda:
        raise ValueError(f"a is not on a device")

    if a.dim() != 2:
        raise ValueError(f"a matrix is not two dimensional")
    
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"a matrix must be square")

    if a.dtype != torch.float32 and a.dtype != torch.float64:
        raise ValueError(f"a matrix must be float32 or float64")
    
    N = a.shape[0]

    # cusolver is 1 indexed
    il_one_indexed = 0
    iu_one_indexed = 0
    eigenvalue_range = False
    total_eigenvalues = N

    if subset_by_index is not None:
        il, iu = subset_by_index
        #TODO: test scipy on ranges and see what they do
        if il > iu or il < 0 or il > N or iu < 0 or iu > N:
            raise ValueError(f"eigval index range is out of bounds")
        
        il_one_indexed = il + 1
        iu_one_indexed = iu + 1
        eigenvalue_range = True
        total_eigenvalues = iu_one_indexed - il_one_indexed + 1
        
    if not overwrite_a:
        out = a.clone()
    else:
        out = a

    w = torch.zeros(a.size(0), dtype=a.dtype, device=a.device)
    info = torch.scalar_tensor(-1, device=a.device, dtype=torch.int)

    # cusolver and cublas are column_major not row_major
    is_upper_triangle_column_major = lower

    stream = cuda.current_stream()

    SingletonClass().kernel.cusolverDnXsyevdx_export(
        out, 
        w, 
        info, 
        il_one_indexed, 
        iu_one_indexed, 
        is_upper_triangle_column_major, 
        eigvals_only, 
        eigenvalue_range, 
        stream.cuda_stream
    )

    cpu_info = info.cpu()
    if cpu_info != 0:
        raise ValueError(f"info of syevdx is not equal to 0: {cpu_info}. Check cusolver docs")
    
    if eigenvalue_range:
        w = w[: total_eigenvalues].clone()
        if eigvals_only:
            return w
        else:
            out = out[: total_eigenvalues].T
            return w, out
    else:
        if eigvals_only:
            return w
        else:
            return w, out

def mgSyevd(a, overwrite_a = False):
    N = a.size(0)
    d = torch.zeros(N, dtype=a.dtype, device=a.device)
    if not overwrite_a:
        out = a.clone()
    else:
        out = a
    SingletonClass().kernel.cusolverMgSyevd_export(out, d)
    return d, out.T