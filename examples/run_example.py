import torch
import scipy as sp

from kernel_tools import warmup, mgSyevd, syevdx
import kernel_tools.kernels as kernels

torch.set_printoptions(linewidth=240)

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

warmup()
# a = torch.arange(10)

N = 5
D = 2
num_eigs = 2

a = torch.randn(N, D, dtype=torch.float64)

kernel_mat = kernel_fn(a,a)
print(kernel_mat)

num_samples = N
top_q_eig = num_eigs

cuda_sub_start = N - num_eigs
cuda_sub_end = N

cuda_total = cuda_sub_end - cuda_sub_start + 1
print(cuda_total)

scipy_sub_start = num_samples - top_q_eig - 1
scipy_sub_end = num_samples - 1

scipy_total = scipy_sub_end - scipy_sub_start + 1
print(scipy_total)

d = torch.zeros(N, dtype=kernel_mat.dtype)
# mgSyevd(kernel_mat, d)
kernel_mat_cuda = kernel_mat.to(device='cuda')
cuda_eigenvalues = syevdx(kernel_mat_cuda, cuda_sub_start, cuda_sub_end)
print(cuda_eigenvalues)

# cuda_eigenvalues, cuda_eigenvectors = syevdx(kernel_mat.to(dtype=torch.float32), cuda_sub_start, cuda_sub_end)
# print(cuda_eigenvalues)
print(kernel_mat_cuda[: cuda_total].T)

scipy_eigenvalues, scipy_eigenvectors = sp.linalg.eigh(kernel_mat.cpu(), subset_by_index=(scipy_sub_start, scipy_sub_end))
print(scipy_eigenvectors)
print(scipy_eigenvalues)
# print(kernel_mat)
# print(d)