import torch

from kernel_tools import warmup, tensor_test, mgSyevd, syevdx
import kernel_tools.kernels as kernels

torch.set_printoptions(linewidth=240)

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

warmup()
a = torch.arange(10)
tensor_test(a)

N = 6
D = 2

a = torch.randn(N, D)

kernel_mat = kernel_fn(a,a)
print(kernel_mat)


d = torch.zeros(N, dtype=kernel_mat.dtype)
# mgSyevd(kernel_mat, d)
kernel_mat = kernel_mat.to(device='cuda')
_, d = syevdx(kernel_mat, num_eigs=1, out=kernel_mat)
print(kernel_mat)
print(d)