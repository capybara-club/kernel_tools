import torch
import time

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import *


if __name__ == "__main__":

    # Can show useful info like:
    # * How many devices the cpp code sees
    # * Workspace requirement of the run
    # * Steps successfully completed before a potential crash
    verbose = False

    # This makes the function write the eigenvectors in to the input matrix.
    # This is the default behavior of the function. The PyTorch side code makes a clone
    # if this is False.
    overwrite_a = True

    kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

    # IMO it is not worth it to try torch.float32, user beware!
    dtype = torch.float64

    # Setting this to None will check for visible devices and use that count
    num_devices = None 

    N = 5_000
    D = 2

    # Calls in to cusolver_mg are really slow due to the need to spawn a process
    # and a new cuda context and a new cusolverMgHandle_t..
    # Should be a small overhead for large N.
    workspace_num_bytes = cusolver_mg_eigh_workspace_requirements(N, dtype, num_devices=num_devices)
    print(f'cusolver_mg_eigh workspace requirement: {workspace_num_bytes // 1_000_000}MB')

    a = torch.randn(N, D, dtype=dtype)

    # Can compute on 'cuda' but matrix needs to be on cpu when run for cusolver_mg_eigh
    kernel_mat = kernel_fn(a,a)

    eigenvectors_mg = kernel_mat.clone()

    cuda_start = time.time()
    cuda_mg_eigenvalues, cuda_mg_eigenvectors = cusolver_mg_eigh(eigenvectors_mg, overwrite_a=overwrite_a, verbose=verbose)
    cuda_end = time.time()
    reconstructed = (cuda_mg_eigenvectors @ torch.diag(cuda_mg_eigenvalues)) @ cuda_mg_eigenvectors.T
    error = torch.norm(reconstructed - kernel_mat)
    print(f'cusolver_mg_eigh error: \t{error}')

    kernel_mat_scipy = kernel_mat.clone()

    scipy_start = time.time()
    scipy_eigenvalues, scipy_eigenvectors = scipy_eigh(kernel_mat_scipy, overwrite_a=overwrite_a)
    scipy_eigenvectors = torch.from_numpy(scipy_eigenvectors)
    scipy_eigenvalues = torch.from_numpy(scipy_eigenvalues)
    scipy_end = time.time()
    reconstructed = (scipy_eigenvectors @ torch.diag(scipy_eigenvalues)) @ scipy_eigenvectors.T
    error = torch.norm(reconstructed - kernel_mat)

    print(f'scipy_eigh error: \t\t{error}')

    print(f'cusolver_mg_eigh time: \t{(cuda_end - cuda_start):.2f} seconds')
    print(f'scipy time: \t\t{(scipy_end - scipy_start):.2f} seconds')
