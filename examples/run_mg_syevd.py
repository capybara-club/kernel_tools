import torch
import time
import os

from scipy.linalg import eigh as scipy_eigh

import kernel_tools.kernels as kernels
from kernel_tools.linalg import *

import multiprocessing as mp

def initialize_cuda_in_subprocess(kernel_mat):
    try:
        print(f"Subprocess PID: {os.getpid()}")
        # torch.cuda.set_device(0)
        cuda_mg_eigenvalues, cuda_mg_eigenvectors = cusolver_mg_eigh(kernel_mat, overwrite_a=True, verbose=True)
        print(f"CUDA initialized in subprocess with PID {os.getpid()}")
        return cuda_mg_eigenvalues, cuda_mg_eigenvectors
    except Exception as e:
        print(f"Failed to initialize CUDA in subprocess: {e}")

def run():
    # Ensure CUDA is not initialized in the main process
    print(f"Main process PID: {os.getpid()}")

    kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)
    dtype = torch.float64

    N = 10000
    D = 2


    a = torch.randn(N, D, dtype=torch.float64)

    kernel_mat = kernel_fn(a,a)
    
    # Create a subprocess to initialize CUDA
    p = mp.Process(target=initialize_cuda_in_subprocess, args=(kernel_mat,))
    p.start()
    p.join()

    print(f"CUDA available in main process: {torch.cuda.is_available()}")

if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    # run()
    # run()

    verbose = False
    overwrite_a = True

    kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)
    dtype = torch.float64

    N = 100
    D = 2

    workspace_num_bytes = cusolver_mg_eigh_workspace_requirements(N, dtype, 4)
    print(workspace_num_bytes)

    workspace_num_bytes = cusolver_mg_eigh_workspace_requirements(1000, dtype, 4)
    print(workspace_num_bytes)

    workspace_num_bytes = cusolver_mg_eigh_workspace_requirements(10000, dtype, 4)
    print(workspace_num_bytes)

    a = torch.randn(N, D, dtype=torch.float64)

    kernel_mat = kernel_fn(a,a)

    cuda_mg_eigenvalues, cuda_mg_eigenvectors = cusolver_mg_eigh(kernel_mat, overwrite_a=overwrite_a, verbose=verbose)

    a = torch.randn(N, D, dtype=torch.float64)

    kernel_mat = kernel_fn(a,a)

    cuda_mg_eigenvalues, cuda_mg_eigenvectors = cusolver_mg_eigh(kernel_mat, overwrite_a=overwrite_a, verbose=verbose)
