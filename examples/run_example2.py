import torch
import time

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import cusolver_eigh, cusolver_mg_eigh

# torch.set_printoptions(linewidth=240)

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)

N = 20000
D = 2

a = torch.randn(N, D, dtype=torch.float64)

kernel_mat = kernel_fn(a,a)
overwrite_a = True


kernel_mat_mg = kernel_mat.clone()
start = time.time()
cuda_mg_eigenvalues, cuda_mg_eigenvectors = cusolver_mg_eigh(kernel_mat_mg, overwrite_a=overwrite_a)
end = time.time()
print(end - start)
print(cuda_mg_eigenvalues)
print(cuda_mg_eigenvectors)

kernel_mat_cpu = kernel_mat.clone()

# start = time.time()
# scipy_eigenvalues, scipy_eigenvectors = scipy_eigh(kernel_mat_cpu, overwrite_a=overwrite_a)
# end = time.time()
# print('scipy eigenvalues:')
# print(scipy_eigenvalues)
# print('scipy eigenvectors:')
# print(scipy_eigenvectors)
# print(end - start)