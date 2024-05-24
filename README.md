# kernel_tools
Tools for kernel methods for PyTorch and Cuda devices

## Linalg
Exports two cusolver functions, cusolverSyevdx and cusolverMgsyevd.

### cusolverdnxsyevdx

cuSolver Documentation: [cusolverDnXsyevdx](https://docs.nvidia.com/cuda/cusolver/#dense-eigenvalue-solver-reference-64-bit-api)

This function extracts eigenvalues/eigenvectors by an index. This can extract the top few
eigenvalues/eigenvectors of a system. Function runs on one Nvidia device.

This function is available in `kernel_tools.linalg.cusolver_eigh`

The function to estimate workspace size for a future run is `kernel_tools.linalg.cusolver_eigh_workspace_requirements`

### cusolverMgsyevd

cuSolver Documentation: [cusolverMgSyevd](https://docs.nvidia.com/cuda/cusolver/#cusolvermgsyevd)

This function extracts all eigenvalues/eigenvectors of a system. This function uses all
visible Nvidia devices to split the work and share data across NVlink if available.

This function is available in `kernel_tools.linalg.cusolver_mg_eigh`

The function to estimate workspace size for a future run is `kernel_tools.linalg.cusolver_mg_eigh_workspace_requirements`

#### WARNING:
There is currently an issue with reusing the `cusolverMgHandle_t` for multiple runs in the same context. It seems that even destroying the handle and creating a new one has the same issue. The only workaround appears to be leaking memory by creating a new handle for each run and leaving the old ones stagnant. A bug report was filed to Nvidia for this.

## Environment Variables

use `export USE_KERNEL_TOOLS_JIT=1` to force use of cpp/cu in JIT mode for development.

use `export USE_KERNEL_TOOLS_JIT=0` or don't export this symbol to fall back to compiled versions.

use `export USE_KERNEL_TOOLS_DEBUG=0` or don't export this symbol to compile cxx and cuda in release mode.

use `export USE_KERNEL_TOOLS_DEBUG=1` to compile cxx and cuda with no optimization. This should compile faster.

use `export USE_KERNEL_TOOLS_VERBOSE=1` to compile cxx and cuda verbose during JIT compilation. This shows your standard compile errors with line numbers in the cpp and cuda files.

use `export USE_KERNEL_TOOLS_VERBOSE=0` or don't export to silence the JIT compilation log lines.