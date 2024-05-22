import torch
import scipy as sp
import time

import kernel_tools.kernels as kernels
from kernel_tools.linalg import cusolver_eigh

torch.set_printoptions(linewidth=240)

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

# warmup()
# a = torch.arange(10)

N = 20000
D = 2
num_eigs = 100

a = torch.randn(N, D, dtype=torch.float32)

kernel_mat = kernel_fn(a,a)
print(kernel_mat)

num_samples = N
top_q_eig = num_eigs

scipy_sub_start = num_samples - top_q_eig - 1
scipy_sub_end = num_samples - 1

subset_by_index = (scipy_sub_start, scipy_sub_end)
eigvals_only = False
overwrite_a = True

d = torch.zeros(N, dtype=kernel_mat.dtype)
# mgSyevd(kernel_mat, d)
kernel_mat_cuda = kernel_mat.to(device='cuda')
# cuda_eigenvalues = syevdx(kernel_mat_cuda, cuda_sub_start, cuda_sub_end)
# print(cuda_eigenvalues)

# cuda_eigenvalues, cuda_eigenvectors = syevdx(kernel_mat.to(dtype=torch.float32), cuda_sub_start, cuda_sub_end)
# print(cuda_eigenvalues)

# print(kernel_mat_cuda)
start = time.time()
cuda_eigenvalues, cuda_eigenvectors = cusolver_eigh(kernel_mat_cuda.to(dtype=torch.float32), subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
print(cuda_eigenvalues)
end = time.time()
print(end - start)

# print(cuda_eigenvectors)

# print(kernel_mat_cuda)
kernel_mat_cpu = kernel_mat.cpu()
start = time.time()
scipy_eigenvalues, scipy_eigenvectors = sp.linalg.eigh(kernel_mat_cpu, subset_by_index=subset_by_index, eigvals_only=eigvals_only, overwrite_a=overwrite_a)
print(scipy_eigenvalues)
end = time.time()
print(end - start)
# print(scipy_eigenvectors)
# print(kernel_mat_cpu)
# print(kernel_mat)
# print(d)