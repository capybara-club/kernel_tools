import torch
import time

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import cusolver_eigh

# torch.set_printoptions(linewidth=240)

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

N = 2000
D = 2
top_q_eig = 100

a = torch.randn(N, D, dtype=torch.float64)

kernel_mat = kernel_fn(a,a)

sub_start = N - top_q_eig - 1
sub_end = N - 1

subset_by_index = (sub_start, sub_end)
eigvals_only = False
overwrite_a = True

kernel_mat_cuda = kernel_mat.to(device='cuda')
kernel_mat_cpu = kernel_mat.cpu()

start = time.time()
cuda_eigenvalues, cuda_eigenvectors = cusolver_eigh(kernel_mat_cuda, subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
end = time.time()
print('cuda eigenvalues:')
print(cuda_eigenvalues)
print('cuda eigenvectors:')
print(cuda_eigenvectors)
print(end - start)

start = time.time()
scipy_eigenvalues, scipy_eigenvectors = scipy_eigh(kernel_mat_cpu, subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
end = time.time()
print('scipy eigenvalues:')
print(scipy_eigenvalues)
print('scipy eigenvectors:')
print(scipy_eigenvectors)
print(end - start)