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

    # on consumer cards float32 is much much faster than float64. On A100, H100, B100 float64 should be about double the time.
    dtype = torch.float32

    N = 5_000
    D = 2
    batch_size = 10

    workspace_num_bytes_device, workspace_num_bytes_host = cusolver_batched_eigh_workspace_requirements(N, batch_size, dtype)
    print(f'cusolver_batched_eigh workspace requirement device: {workspace_num_bytes_device // 1_000_000}MB')
    print(f'cusolver_batched_eigh workspace requirement host: {workspace_num_bytes_host // 1_000_000}MB')

    a = torch.randn(N, D, dtype=dtype)

    # Leaving on cpu so can give to scipy on cpu
    kernel_mat = kernel_fn(a,a)

    # Needs to be on a device to call.
    eigenvectors_matrix = kernel_mat.clone().cuda()
    eigenvectors_matrix = eigenvectors_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

    # eigenvectors_matrix is now (batch_size, N, N)
    # uses repeat so its a full memory chunk rather than a replicated view

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    cuda_eigenvalues, cuda_eigenvectors = cusolver_batched_eigh(eigenvectors_matrix, overwrite_a=overwrite_a, verbose=verbose)
    end.record()
    torch.cuda.synchronize()

    cuda_eigenvectors = cuda_eigenvectors.cpu()
    cuda_eigenvalues = cuda_eigenvalues.cpu()
    batched_diag = torch.diag_embed(cuda_eigenvalues)

    # Now you can do the matrix multiplication
    # bmm is batch matrix multiply
    reconstructed = torch.bmm(cuda_eigenvectors, batched_diag)
    reconstructed = torch.bmm(reconstructed, cuda_eigenvectors.transpose(1, 2))
    # reconstructed = (cuda_eigenvectors @ torch.diag(cuda_eigenvalues)) @ cuda_eigenvectors.T
    error = torch.norm(reconstructed - kernel_mat)
    print(f'cusolver_batched_eigh error: \t{error}')

    kernel_mat_scipy = kernel_mat.clone()

    scipy_start = time.time()
    scipy_eigenvalues, scipy_eigenvectors = scipy_eigh(kernel_mat_scipy, overwrite_a=overwrite_a)
    scipy_eigenvectors = torch.from_numpy(scipy_eigenvectors)
    scipy_eigenvalues = torch.from_numpy(scipy_eigenvalues)
    scipy_end = time.time()
    reconstructed = (scipy_eigenvectors @ torch.diag(scipy_eigenvalues)) @ scipy_eigenvectors.T
    error = torch.norm(reconstructed - kernel_mat)

    print(f'scipy_eigh error: \t{error}')

    time_elapsed = start.elapsed_time(end)
    print(f'cusolver_batched_eigh time (for {batch_size} matrices): \t{(time_elapsed/1000):.2f} seconds')
    print(f'scipy time (for 1 matrix): \t\t{(scipy_end - scipy_start):.2f} seconds')