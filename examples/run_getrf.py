import torch
import time

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import *

if __name__ == "__main__":
    verbose = False

    # These make the cusolver functions behave as-is. They overwrite the information
    # in to the input tensor to save duplication
    # If overwrite is False for any of these then the python code clones the input tensor first
    overwrite_a = True
    overwrite_targets = True

    kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

    dtype = torch.float64

    N = 8
    D = 4

    nrhs = 3

    workspace_num_bytes_device, workspace_num_bytes_host = cusolver_getrf_workspace_requirements(N, D, dtype)
    print(f'cusolver_eigh workspace requirement device: {workspace_num_bytes_device // 1_000_000}MB')
    print(f'cusolver_eigh workspace requirement host: {workspace_num_bytes_host // 1_000_000}MB')

    a = torch.randn(N, D, dtype=dtype)

    # Generate square kernel matrix
    kernel_mat = kernel_fn(a,a)

    # Pytorch is row major but cusolver expects targets to be ldN * nrhs. so nrhs must be number of rows
    targets = torch.randn(nrhs, N, dtype=dtype)

    cuda_kernel_matrix = kernel_mat.clone().cuda()
    cuda_targets = targets.clone().cuda()

    cuda_factor_start = time.time()
    # ipiv are the pivots that were generated from the getrf internals. See cusolver docs.
    cuda_factored, cuda_ipiv = cusolver_getrf(cuda_kernel_matrix, overwrite_a=overwrite_a, verbose=verbose)
    cuda_factor_end = time.time()
    
    print(f'cusolver_getrf time: \t{(cuda_factor_start - cuda_factor_end):.2f} seconds')
    
    # Ensure that the tensor returned is the same tensor if overwrite is true
    kernel_matrix_is_factored_matrix = cuda_kernel_matrix.data_ptr() == cuda_factored.data_ptr()
    if overwrite_a:
        assert kernel_matrix_is_factored_matrix
    else:
        assert not kernel_matrix_is_factored_matrix

    cuda_solution = cusolver_getrs(cuda_factored, cuda_ipiv, cuda_targets, overwrite_targets=overwrite_targets, verbose=verbose)

    # Ensure that the tensor returned is the same tensor if overwrite is true
    targets_is_solution = cuda_targets.data_ptr() == cuda_solution.data_ptr()
    if overwrite_targets:
        assert targets_is_solution
    else:
        assert not targets_is_solution

    assert(kernel_mat.cuda() @ cuda_solution.T, targets.T)
