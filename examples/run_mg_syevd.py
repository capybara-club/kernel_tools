import torch
import time

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import *

kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)
dtype = torch.float64

N = 10000
D = 2

a = torch.randn(N, D, dtype=torch.float64)

kernel_mat = kernel_fn(a,a)

cuda_mg_eigenvalues, cuda_mg_eigenvectors = cusolver_mg_eigh(kernel_mat, overwrite_a=True, verbose=True)


a = torch.randn(N, D, dtype=torch.float64)

kernel_mat = kernel_fn(a,a)

cuda_mg_eigenvalues, cuda_mg_eigenvectors = cusolver_mg_eigh(kernel_mat, overwrite_a=True, verbose=True)
