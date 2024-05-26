import torch
import torch.cuda as cuda
import time
import cpuinfo

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import *

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

D = 2

def profile_cuda(N, num_eigs, dtype):
    a = torch.randn(N, D, dtype=dtype)
    kernel_mat = kernel_fn(a,a)
    sub_start = N - num_eigs - 1
    sub_end = N - 1
    subset_by_index = (sub_start, sub_end)
    eigvals_only = False
    overwrite_a = True
    
    workspace_device, workspace_host = cusolver_eigh_workspace_requirements(N, dtype)
    try:
        cuda.empty_cache()
        kernel_mat_cuda = kernel_mat.to(device='cuda')
        start = time.time()
        cuda_eigenvalues, cuda_eigenvectors = cusolver_eigh(kernel_mat_cuda, subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
        end = time.time()
        return (end - start), workspace_device
    except:
        return None, workspace_device

def profile_cuda_mg(N, dtype, max_num_devices, verbose=False):
    a = torch.randn(N, D, dtype=dtype)
    kernel_mat = kernel_fn(a,a)
    overwrite_a = True
    kernel_mat_mg = kernel_mat
    
    workspace_bytes = None
    try:
        cuda.empty_cache()
        workspace_bytes = cusolver_mg_eigh_workspace_requirements(N, dtype, max_num_devices)
        start = time.time()
        cuda_eigenvalues, cuda_eigenvectors = cusolver_mg_eigh(kernel_mat_mg, overwrite_a=overwrite_a, max_num_devices=max_num_devices, verbose=verbose)
        end = time.time()
        return (end - start), workspace_bytes
    except Exception as e:
        # print(f'{e}')
        return None, workspace_bytes
    
def print_stats(
    N, num_eigs, dtype, run_time, device_name, method_name, workspace_bytes
):
    print(f'N:{N}, num_eigs:{num_eigs}, dtype:{dtype}, run_time:{run_time}, device:{device_name}, method:{method_name}, workspace_bytes:{workspace_bytes}')

def profile_scipy(N, num_eigs, dtype):
    # dont even try
    if N > 35_000:
        return None
    a = torch.randn(N, D, dtype=dtype)
    kernel_mat = kernel_fn(a,a)
    sub_start = N - num_eigs - 1
    sub_end = N - 1
    subset_by_index = (sub_start, sub_end)
    eigvals_only = False
    overwrite_a = True
    kernel_mat_cpu = kernel_mat.cpu()

    start = time.time()
    scipy_eigenvalues, scipy_eigenvectors = scipy_eigh(kernel_mat_cpu, subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
    end = time.time()
    return end - start

if __name__ == "__main__":
    n_sizes = [15_000, 20_000, 25_000]
    # n_sizes = [100, 1_000, 2_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000]
    # n_sizes = [100, 1_000, 2_000, 5_000, 10_000, ]
    n_eigs =  [ 10,    50,   100,   200,    500,    750,  1_000,  2_000,  3_000,  4_000,  5_000,  7_000,  9_000]
    # n_dtypes = [torch.float32, torch.float64]
    n_dtypes = [torch.float32]

    should_profile_cuda = True
    should_profile_cuda_mg = False
    should_profile_scipy = False

    if should_profile_scipy:
        method_name = 'scipy.linalg.eigh'
        device_name = cpuinfo.get_cpu_info()['brand_raw']
        for dtype in n_dtypes:
            for N, num_eigs in list(zip(n_sizes, n_eigs)):
                run_time = profile_scipy(N, num_eigs, dtype)
                print_stats(N, num_eigs, dtype, run_time, device_name, method_name, None)

    if should_profile_cuda:
        device_name = cuda.get_device_name(0)
        method_name = 'cusolverSyevdx'
        for dtype in n_dtypes:
            for N, num_eigs in list(zip(n_sizes, n_eigs)):
                run_time, workspace_bytes = profile_cuda(N, num_eigs, dtype)
                print_stats(N, num_eigs, dtype, run_time, device_name, method_name, workspace_bytes)

    if should_profile_cuda_mg:
        for num_devices in range(1, cuda.device_count() + 1):
            device_names = []
            for device_num in range(num_devices):
                device_name = cuda.get_device_name(device_num)
                device_names.append(device_name)

            device_names_str = ', '.join(device_names)
            method_name = 'cusolverMgsyevd'
            workspace_bytes = None
            verbose = False
            for dtype in n_dtypes:
                for N in n_sizes:
                    run_time, workspace_bytes = profile_cuda_mg(N, dtype, num_devices, verbose=verbose)
                    print_stats(N, N, dtype, run_time, device_names_str, method_name, workspace_bytes)
