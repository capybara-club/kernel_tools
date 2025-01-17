from torch.utils.cpp_extension import load, load_inline
import os
import sys
import subprocess
import torch
import torch.cuda as cuda
import multiprocessing as mp

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
USE_VERBOSE = os.getenv('USE_KERNEL_TOOLS_VERBOSE', '0') == '1'

@singleton
class SingletonClass:
    def __init__(self):
        try:
            # Installation via pip
            if USE_JIT:
                if USE_VERBOSE:
                    print('Running kernel_tools JIT (first run can take a minute)')
                kernel = load_jit()
            else:
                try:
                    if USE_VERBOSE:
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
        overwrite_a,
        subset_by_index, 
        lower, 
        eigvals_only,
        verbose
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
        stream.cuda_stream,
        verbose
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
            return w, out.T
        
def syevdx_workspace_query(
        N,
        dtype
):
    if dtype != torch.float32 and dtype != torch.float64:
        raise ValueError("Unsupported dtype")
    is_fp32 = dtype == torch.float32
    workspaceBytesDeviceTensor = torch.tensor(0, dtype=torch.uint64)
    workspaceBytesHostTensor = torch.tensor(0, dtype=torch.uint64)
    SingletonClass().kernel.cusolverDnXsyevdx_workspace_query_export(
        N,
        is_fp32,
        workspaceBytesDeviceTensor,
        workspaceBytesHostTensor
    )
    return workspaceBytesDeviceTensor.item(), workspaceBytesHostTensor.item()

def syev_batched(
        a,
        overwrite_a,
        lower, 
        eigvals_only,
        verbose
):

    if not a.is_cuda:
        raise ValueError(f"batched a is not on a device")

    if a.dim() != 3:
        raise ValueError(f"batched a matrix is not three dimensional")
    
    if a.shape[1] != a.shape[2]:
        raise ValueError(f"batched a matrix must be square")
    
    # if a.shape[0] != 1:
    #     raise ValueError(f"batched a matrix must be dimension (1,X,X) for now")

    if a.dtype != torch.float32 and a.dtype != torch.float64:
        raise ValueError(f"batched a matrix must be float32 or float64")
    
    N = a.shape[1]
    batch_size = a.shape[0]
        
    if not overwrite_a:
        out = a.clone()
    else:
        out = a

    w = torch.zeros(batch_size, N, dtype=a.dtype, device=a.device)
    info = torch.zeros(batch_size, device=a.device, dtype=torch.int)

    # cusolver and cublas are column_major not row_major
    is_upper_triangle_column_major = lower

    stream = cuda.current_stream()

    SingletonClass().kernel.cusolverDnXsyev_batched_export(
        out, 
        w, 
        info, 
        is_upper_triangle_column_major, 
        eigvals_only, 
        stream.cuda_stream,
        verbose
    )

    cpu_info = info.cpu()
    if not (cpu_info == 0).all():
        raise ValueError(f"info of syev_batched is not equal to 0: {cpu_info}. Check cusolver docs")
    
    if eigvals_only:
        return w
    else:
        return w, out.transpose(1,2)

def syev_batched_workspace_query(
        N,
        batch_size,
        dtype
):
    if dtype != torch.float32 and dtype != torch.float64:
        raise ValueError("Unsupported dtype")
    is_fp32 = dtype == torch.float32
    workspaceBytesDeviceTensor = torch.tensor(0, dtype=torch.uint64)
    workspaceBytesHostTensor = torch.tensor(0, dtype=torch.uint64)
    SingletonClass().kernel.cusolverDnXsyev_batched_workspace_query_export(
        N,
        batch_size,
        is_fp32,
        workspaceBytesDeviceTensor,
        workspaceBytesHostTensor
    )
    return workspaceBytesDeviceTensor.item(), workspaceBytesHostTensor.item()

def call_mgSyevd(queue, out, d, max_num_devices, verbose):
    try:
        SingletonClass().kernel.cusolverMgSyevd_export(out, d, max_num_devices, verbose)
    except Exception as e:
        queue.put(e)

def mgSyevd(a, overwrite_a = False, max_num_devices=16, verbose = False):
    # This function allocates outside of PyTorch, should release as much vram as possible before call.
    cuda.empty_cache()
    N = a.size(0)
    d = torch.zeros(N, dtype=a.dtype, device=a.device)
    if not overwrite_a:
        out = a.clone()
    else:
        out = a

    # A cusolverMg bug made this insane. The cusolverMgHandle being reused will have 
    # cusolverMgSyevd fail each subsequent call. If you destroy the handle and create a new
    # one this issue somehow persists, even if the handle only had device select called on it. 
    # The only other way is to create a new handle and leak 
    # a somewhat substantial amount of vram for each handle. This was enough of a leak
    # such that benchmarking increasily larger problem sizes made certain problem sizes impossible, 
    # where individual runs would successfully complete. Apparently if you create a spawned process with its own
    # cuda context, when that process died, all the memory is cleaned up and the 
    # cusolverMgHandle issue is gone. I filed a bug report with Nvidia and I'm waiting to 
    # hear back. However this would force any user to be using the latest cuda toolkit
    # version after patch.
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    queue = mp.Queue()

    p = mp.Process(target=call_mgSyevd, args=(queue, out, d, max_num_devices, verbose))
    p.start()
    p.join()

    if not queue.empty():
        error = queue.get()
        raise ValueError(f'mgSyevd exception:{error}')

    return d, out.T

def call_mgSyevd_workspace_query(N, num_devices, is_fp32, use_num_devices_visible, workspaceNumElementsTensor, verbose):
    SingletonClass().kernel.cusolverMgSyevd_workspace_query_export(
        N,
        num_devices,
        is_fp32,
        use_num_devices_visible,
        workspaceNumElementsTensor,
        verbose
    )

def mgSyevd_workspace_query(
        N,
        num_devices,
        dtype,
        verbose
):
    if dtype != torch.float32 and dtype != torch.float64:
        raise ValueError("Unsupported dtype")
    
    use_num_devices_visible = False
    if num_devices is None:
        num_devices = 0
        use_num_devices_visible = True
        
    is_fp32 = dtype == torch.float32
    workspaceNumElementsTensor = torch.tensor(0, dtype=torch.int64)

    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    p = mp.Process(target=call_mgSyevd_workspace_query, args=(N, num_devices, is_fp32, use_num_devices_visible, workspaceNumElementsTensor, verbose))
    p.start()
    p.join()

    return workspaceNumElementsTensor.item() * dtype.itemsize