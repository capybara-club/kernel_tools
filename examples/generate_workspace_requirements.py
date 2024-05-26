import torch

from kernel_tools.linalg import *

if __name__ == "__main__":
    dtype = torch.float64

    print('N, cusolver_eigh, cusolver_mg_eigh(1 gpu), cusolver_mg_eigh(2 gpu), cusolver_mg_eigh(4 gpu), cusolver_mg_eigh(8 gpu),')
    sizes = [1_000, 2_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]
    for N in sizes:
        print(f'{N}, ', end='')
        workspace_bytes, _ = cusolver_eigh_workspace_requirements(N, dtype)
        print(f'{workspace_bytes // 1_000_000}, ', end='')

        num_gpus = [1,2,4,8]
        for num_gpu in num_gpus:
            workspace_bytes = cusolver_mg_eigh_workspace_requirements(N, dtype, num_gpu)
            print(f'{workspace_bytes // 1_000_000}, ', end='')
        print('')