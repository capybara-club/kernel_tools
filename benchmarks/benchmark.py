import torch
import time

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import cusolver_eigh

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

def profile(N, num_eigs, dtype):
    D = 2
    a = torch.randn(N, D, dtype=dtype)
    kernel_mat = kernel_fn(a,a)
    sub_start = N - num_eigs - 1
    sub_end = N - 1
    subset_by_index = (sub_start, sub_end)
    eigvals_only = False
    overwrite_a = True
    kernel_mat_cuda = kernel_mat.to(device='cuda')
    kernel_mat_cpu = kernel_mat.cpu()
    cuda_time = None
    scipy_time = None

    try:
        start = time.time()
        cuda_eigenvalues, cuda_eigenvectors = cusolver_eigh(kernel_mat_cuda, subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
        end = time.time()
        cuda_time = end - start
    except:
        print('out of memory')

    start = time.time()
    scipy_eigenvalues, scipy_eigenvectors = scipy_eigh(kernel_mat_cpu, subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
    end = time.time()
    scipy_time = end - start

    return cuda_time, scipy_time

n_sizes = [100, 1_000, 2_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
n_eigs =  [ 10,    50,   100,   200,    500,    750,  1_000,  2_000, 3_000]
n_dtypes = [torch.float32, torch.float64]

for dtype in n_dtypes:
    for N, num_eigs in list(zip(n_sizes, n_eigs)):
        cuda_time, scipy_time = profile(N, num_eigs, dtype)

        print(f'N:{N}, num_eigs:{num_eigs}, dtype:{dtype}, cuda_time:{cuda_time}, scipy_time:{scipy_time}')
        # print(cuda_time)
        # print(scipy_time)