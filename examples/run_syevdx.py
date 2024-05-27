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

    N = 5_000
    D = 2

    workspace_num_bytes_device, workspace_num_bytes_host = cusolver_eigh_workspace_requirements(N, dtype)
    print(f'cusolver_eigh workspace requirement device: {workspace_num_bytes_device // 1_000_000}MB')
    print(f'cusolver_eigh workspace requirement host: {workspace_num_bytes_host // 1_000_000}MB')

    a = torch.randn(N, D, dtype=dtype)

    # Leaving on cpu so can give to scipy on cpu
    kernel_mat = kernel_fn(a,a)

    # Needs to be on a device to call.
    eigenvectors_matrix = kernel_mat.clone().cuda()

    cuda_start = time.time()
    cuda_eigenvalues, cuda_eigenvectors = cusolver_eigh(eigenvectors_matrix, overwrite_a=overwrite_a, verbose=verbose)
    cuda_eigenvectors = cuda_eigenvectors.cpu()
    cuda_eigenvalues = cuda_eigenvalues.cpu()
    cuda_end = time.time()
    reconstructed = (cuda_eigenvectors @ torch.diag(cuda_eigenvalues)) @ cuda_eigenvectors.T
    error = torch.norm(reconstructed - kernel_mat)
    print(f'cusolver_eigh error: \t{error}')

    kernel_mat_scipy = kernel_mat.clone()

    scipy_start = time.time()
    scipy_eigenvalues, scipy_eigenvectors = scipy_eigh(kernel_mat_scipy, overwrite_a=overwrite_a)
    scipy_eigenvectors = torch.from_numpy(scipy_eigenvectors)
    scipy_eigenvalues = torch.from_numpy(scipy_eigenvalues)
    scipy_end = time.time()
    reconstructed = (scipy_eigenvectors @ torch.diag(scipy_eigenvalues)) @ scipy_eigenvectors.T
    error = torch.norm(reconstructed - kernel_mat)

    print(f'scipy_eigh error: \t{error}')

    print(f'cusolver_eigh time: \t{(cuda_end - cuda_start):.2f} seconds')
    print(f'scipy time: \t\t{(scipy_end - scipy_start):.2f} seconds')