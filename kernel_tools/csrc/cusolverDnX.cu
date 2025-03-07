#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <signal.h>

#include "kernel_tools.h"
#include "utils.h"

void *cuda_alloc(size_t size) {
    void *device_data = c10::cuda::CUDACachingAllocator::raw_alloc(size);
    return device_data;
}

void cuda_free(void *device_data) {
    c10::cuda::CUDACachingAllocator::raw_delete(device_data);
}

void *host_alloc(size_t size) {
    auto& host_allocator = *at::getCPUAllocator();
    void *host_data = host_allocator.raw_allocate(size);
    return host_data;
}

void host_free(void *host_data) {
    auto& host_allocator = *at::getCPUAllocator();
    host_allocator.raw_deallocate(host_data);
}

// Class to manage the cuBLAS handle
class CuSolverHandle {
public:
    CuSolverHandle() {
        cusolverStatus_t status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSolver handle");
        }
    }

    ~CuSolverHandle() {
        cusolverStatus_t status = cusolverDnDestroy(handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "Failed to destroy cuSolver handle" << std::endl;
        }
    }

    cusolverDnHandle_t get() const {
        return handle;
    }

private:
    cusolverDnHandle_t handle;
};

// Singleton pattern to ensure only one instance of CuSolverHandle
CuSolverHandle& get_cusolver_dn_handle() {
    static CuSolverHandle handle;
    return handle;
}

template <typename data_type>
void cusolverDnXgetrf_workspace_template(
    cusolverDnHandle_t handle,
    int64_t M,
    int64_t N,
    cudaDataType_t cuda_data_type,
    int64_t lda,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost
) {

    cusolverStatus_t status = cusolverDnXgetrf_bufferSize(
        handle,                     // handle
        NULL,                       // params
        M,                          // M
        N,                          // N
        cuda_data_type,             // dataTypeA
        NULL,                       // A
        lda,                        // lda
        cuda_data_type,             // computeType
        workspaceInBytesOnDevice,  // workspaceInBytesOnDevice
        workspaceInBytesOnHost     // workspaceInBytesOnHost
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXgetrf_bufferSize: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXgetrf_bufferSize");
    }
}

template <typename data_type>
void cusolverDnXgetrf_template(
    torch::Tensor a,
    torch::Tensor ipiv,
    torch::Tensor info,
    uintptr_t stream_ptr,
    cudaDataType_t cuda_data_type,
    bool verbose
) {
    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    cusolverDnSetStream(handle, stream);

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    cusolverStatus_t status;

    cudaDataType data_type_a = cuda_data_type;
    cudaDataType compute_type = cuda_data_type;

    int M = a.size(0);
    int N = a.size(1);

    bool is_a_T;
    int lda;
    tensor_stats(a, &is_a_T, &lda);

    cusolverDnXgetrf_workspace_template<data_type>(
        handle, M, N, cuda_data_type, lda, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
    );

    if (verbose) {
        printf("cusolverDnXgetrf workspace requested: device: %zu bytes, host: %zu bytes\n", workspaceInBytesOnDevice, workspaceInBytesOnHost);
    }

    void *cuda_data = cuda_alloc(workspaceInBytesOnDevice);
    void *host_data = host_alloc(workspaceInBytesOnHost);

    status = cusolverDnXgetrf(
        handle,
        NULL,
        M,
        N,
        data_type_a,
        a.data_ptr<data_type>(),
        lda,
        ipiv.data_ptr<int64_t>(),
        compute_type,
        cuda_data,
        workspaceInBytesOnDevice,
        host_data,
        workspaceInBytesOnHost,
        info.data_ptr<int>()
    );

    cuda_free(cuda_data);
    host_free(host_data);

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXgetrf: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXgetrf");
    }

}

template <typename data_type>
void cusolverDnXgetrs_template(
    torch::Tensor a, 
    torch::Tensor ipiv,
    torch::Tensor b,
    torch::Tensor info,
    uintptr_t stream_ptr,
    cudaDataType_t cuda_data_type,
    bool verbose
) {
    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    cusolverDnSetStream(handle, stream);

    cusolverStatus_t status;

    cudaDataType data_type_a = cuda_data_type;
    cudaDataType data_type_b = cuda_data_type;

    int M = a.size(0);
    int N = a.size(1);

    int nrhs = b.size(1);

    bool is_a_T;
    int lda;
    tensor_stats(a, &is_a_T, &lda);

    bool is_b_T;
    int ldb;
    tensor_stats(b, &is_b_T, &ldb);

    status = cusolverDnXgetrs(
        handle,
        NULL,
        CUBLAS_OP_T,
        N,
        nrhs,
        data_type_a,
        a.data_ptr<data_type>(),
        lda,
        ipiv.data_ptr<int64_t>(),
        data_type_b,
        b.data_ptr<data_type>(),
        ldb,
        info.data_ptr<int>()
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXgetrs: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXgetrs");
    }

}

template <typename data_type>
void cusolverDnXsyevdx_workspace_template(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t N,
    cudaDataType_t cuda_data_type,
    int64_t lda,
    int64_t il,
    int64_t iu,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost
) {

    data_type vl;
    data_type vu;
    int64_t h_meig;

    cusolverStatus_t status = cusolverDnXsyevdx_bufferSize(
        handle,                     // handle
        NULL,                       // params
        jobz,                       // jobz
        range,                      // range
        uplo,                       // uplo
        N,                          // N
        cuda_data_type,             // dataTypeA
        NULL,                       // A
        lda,                        // lda
        &vl,                        // vl
        &vu,                        // vu
        il,                         // il
        iu,                         // iu
        &h_meig,                    // h_meig
        cuda_data_type,             // dataTypeW
        NULL,                       // W
        cuda_data_type,             // computeType
        workspaceInBytesOnDevice,  // workspaceInBytesOnDevice
        workspaceInBytesOnHost     // workspaceInBytesOnHost
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXsyevdx_bufferSize: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXsyevdx_bufferSize");
    }
}

template <typename data_type>
void cusolverDnXsyevdx_template(
    torch::Tensor a, 
    torch::Tensor w,
    torch::Tensor info,
    int il,
    int iu,
    bool upper_triangle,
    bool eigenvalues_only,
    bool eigen_range,
    uintptr_t stream_ptr,
    cudaDataType_t cuda_data_type,
    bool verbose
) {
    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();
    cusolverEigRange_t range = eigen_range ? CUSOLVER_EIG_RANGE_I : CUSOLVER_EIG_RANGE_ALL;
    cublasFillMode_t uplo = upper_triangle ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cusolverEigMode_t jobz = eigenvalues_only ? CUSOLVER_EIG_MODE_NOVECTOR : CUSOLVER_EIG_MODE_VECTOR;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    cusolverDnSetStream(handle, stream);

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    cusolverStatus_t status;

    cudaDataType data_type_a = cuda_data_type;
    cudaDataType data_type_w = cuda_data_type;
    cudaDataType compute_type = cuda_data_type;

    int N = a.size(0);

    bool is_a_T;
    int lda;
    tensor_stats(a, &is_a_T, &lda);

    data_type vl;
    data_type vu;

    int64_t h_meig = 0;

    cusolverDnXsyevdx_workspace_template<data_type>(
        handle, jobz, range, uplo, N, cuda_data_type, lda, il, iu, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
    );

    if (verbose) {
        printf("cusolverDnXsyevdx workspace requested: device: %zu bytes, host: %zu bytes\n", workspaceInBytesOnDevice, workspaceInBytesOnHost);
    }

    void *cuda_data = cuda_alloc(workspaceInBytesOnDevice);
    void *host_data = host_alloc(workspaceInBytesOnHost);

    status = cusolverDnXsyevdx(
        handle,
        NULL,
        jobz,
        range,
        uplo,
        N,
        data_type_a,
        a.data_ptr<data_type>(),
        lda,
        &vl,
        &vu,
        (int64_t)il,
        (int64_t)iu,
        &h_meig,
        data_type_w,
        w.data_ptr<data_type>(),
        compute_type,
        cuda_data,
        workspaceInBytesOnDevice,
        host_data,
        workspaceInBytesOnHost,
        info.data_ptr<int>()
    );

    cuda_free(cuda_data);
    host_free(host_data);

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXsyevdx: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXsyevdx");
    }

}

template <typename data_type>
void cusolverDnXsyev_batched_workspace_template(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t N,
    int64_t batch_size,
    cudaDataType_t cuda_data_type,
    int64_t lda,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost
) {
    cusolverStatus_t status = cusolverDnXsyevBatched_bufferSize(
        handle,                     // handle
        NULL,                       // params
        jobz,                       // jobz
        uplo,                       // uplo
        N,                          // N
        cuda_data_type,             // dataTypeA
        NULL,                       // A
        lda,                        // lda
        cuda_data_type,             // dataTypeW
        NULL,                       // W
        cuda_data_type,             // computeType
        workspaceInBytesOnDevice,  // workspaceInBytesOnDevice
        workspaceInBytesOnHost,     // workspaceInBytesOnHost
        batch_size
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXsyevBatched_bufferSize: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXsyevBatched_bufferSize");
    }
}

template <typename data_type>
void cusolverDnXsyev_batched_template(
    torch::Tensor a, 
    torch::Tensor w,
    torch::Tensor info,
    bool upper_triangle,
    bool eigenvalues_only,
    uintptr_t stream_ptr,
    cudaDataType_t cuda_data_type,
    bool verbose
) {
    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();
    cublasFillMode_t uplo = upper_triangle ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cusolverEigMode_t jobz = eigenvalues_only ? CUSOLVER_EIG_MODE_NOVECTOR : CUSOLVER_EIG_MODE_VECTOR;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    cusolverDnSetStream(handle, stream);

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    cusolverStatus_t status;

    cudaDataType data_type_a = cuda_data_type;
    cudaDataType data_type_w = cuda_data_type;
    cudaDataType compute_type = cuda_data_type;

    int N = a.size(1);
    int batch_size = a.size(0);

    bool is_a_T;
    int lda;
    tensor_stats(a, &is_a_T, &lda);

    cusolverDnXsyev_batched_workspace_template<data_type>(
        handle, jobz, uplo, N, batch_size, cuda_data_type, lda, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
    );

    if (verbose) {
        printf("cusolverDnXsyevBatched workspace requested: device: %zu bytes, host: %zu bytes\n", workspaceInBytesOnDevice, workspaceInBytesOnHost);
    }

    void *cuda_data = cuda_alloc(workspaceInBytesOnDevice);
    void *host_data = host_alloc(workspaceInBytesOnHost);

    status = cusolverDnXsyevBatched(
        handle,
        NULL,
        jobz,
        uplo,
        N,
        data_type_a,
        a.data_ptr<data_type>(),
        lda,
        data_type_w,
        w.data_ptr<data_type>(),
        compute_type,
        cuda_data,
        workspaceInBytesOnDevice,
        host_data,
        workspaceInBytesOnHost,
        info.data_ptr<int>(),
        batch_size
    );

    cuda_free(cuda_data);
    host_free(host_data);

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXsyevdx: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXsyevdx");
    }

}

static void signal_handler(int signum) {
    exit(signum);
}

void cusolverDnXgetrf_export(
    torch::Tensor a, 
    torch::Tensor ipiv,
    torch::Tensor info,
    uintptr_t stream_ptr,
    bool verbose
) {
    signal(SIGINT, signal_handler);
    
    if (a.dtype() == torch::kFloat32) {
        return cusolverDnXgetrf_template<float>(
            a, ipiv, info, stream_ptr, CUDA_R_32F, verbose
        );
    } 

    if (a.dtype() == torch::kFloat64) {
        return cusolverDnXgetrf_template<double>(
            a, ipiv, info, stream_ptr, CUDA_R_64F, verbose
        );
    }

    // If it gets here the dtype isn't supported
    throw std::runtime_error("Tensor needs to have dtype either float32 or float64");
}

void cusolverDnXgetrf_workspace_query_export(
    int M,
    int N,
    bool is_fp32,
    torch::Tensor workspaceBytesDevice,
    torch::Tensor workspaceBytesHost
) {
    signal(SIGINT, signal_handler);

    if (workspaceBytesDevice.dtype() != torch::kUInt64 || workspaceBytesHost.dtype() != torch::kUInt64) {
        throw std::runtime_error("Workspace sizes needs to have dtype uint64");
    }

    if (workspaceBytesDevice.numel() != 1 || workspaceBytesHost.numel() != 1) {
        throw std::runtime_error("Workspace sizes need to have only one element");
    }

    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();
    // because pytorch is row major and cublas is column major, the triangle is flipped

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    int lda = N;

    if (is_fp32) {
        cudaDataType cuda_data_type = CUDA_R_32F;
        cusolverDnXgetrf_workspace_template<float>(
            handle, M, N, cuda_data_type, lda, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
    } else {
        cudaDataType cuda_data_type = CUDA_R_64F;
        cusolverDnXgetrf_workspace_template<double>(
            handle, M, N, cuda_data_type, lda, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
    }

    *workspaceBytesDevice.data_ptr<size_t>() = workspaceInBytesOnDevice;
    *workspaceBytesHost.data_ptr<size_t>() = workspaceInBytesOnHost;
}

void cusolverDnXgetrs_export(
    torch::Tensor a, 
    torch::Tensor ipiv,
    torch::Tensor b,
    torch::Tensor info,
    uintptr_t stream_ptr,
    bool verbose
) {
    signal(SIGINT, signal_handler);
    
    if (a.dtype() == torch::kFloat32) {
        return cusolverDnXgetrs_template<float>(
            a, ipiv, b, info, stream_ptr, CUDA_R_32F, verbose
        );
    } 

    if (a.dtype() == torch::kFloat64) {
        return cusolverDnXgetrs_template<double>(
            a, ipiv, b, info, stream_ptr, CUDA_R_64F, verbose
        );
    }

    // If it gets here the dtype isn't supported
    throw std::runtime_error("Tensor needs to have dtype either float32 or float64");
}

void cusolverDnXsyevdx_export(
    torch::Tensor a, 
    torch::Tensor w,
    torch::Tensor info,
    int il,
    int iu,
    bool upper_triangle,
    bool eigenvalues_only,
    bool eigen_range,
    uintptr_t stream_ptr,
    bool verbose
) {
    signal(SIGINT, signal_handler);
    
    if (a.dtype() == torch::kFloat32) {
        return cusolverDnXsyevdx_template<float>(
            a, w, info, il, iu, upper_triangle, eigenvalues_only, eigen_range, stream_ptr, CUDA_R_32F, verbose
        );
    } 

    if (a.dtype() == torch::kFloat64) {
        return cusolverDnXsyevdx_template<double>(
            a, w, info, il, iu, upper_triangle, eigenvalues_only, eigen_range, stream_ptr, CUDA_R_64F, verbose
        );
    }

    // If it gets here the dtype isn't supported
    throw std::runtime_error("Tensor needs to have dtype either float32 or float64");
}

void cusolverDnXsyevdx_workspace_query_export(
    int N,
    bool is_fp32,
    torch::Tensor workspaceBytesDevice,
    torch::Tensor workspaceBytesHost
) {
    signal(SIGINT, signal_handler);

    if (workspaceBytesDevice.dtype() != torch::kUInt64 || workspaceBytesHost.dtype() != torch::kUInt64) {
        throw std::runtime_error("Workspace sizes needs to have dtype uint64");
    }

    if (workspaceBytesDevice.numel() != 1 || workspaceBytesHost.numel() != 1) {
        throw std::runtime_error("Workspace sizes need to have only one element");
    }

    // These don't appear to affect the workspace sizes
    bool eigen_range = true;
    bool upper_triangle = true;
    bool eigenvalues_only = false;

    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();
    // because pytorch is row major and cublas is column major, the triangle is flipped
    cusolverEigRange_t range = eigen_range ? CUSOLVER_EIG_RANGE_I : CUSOLVER_EIG_RANGE_ALL;
    cublasFillMode_t uplo = upper_triangle ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cusolverEigMode_t jobz = eigenvalues_only ? CUSOLVER_EIG_MODE_NOVECTOR : CUSOLVER_EIG_MODE_VECTOR;

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    int lda = N;
    int il = 0;
    int iu = 0;

    if (is_fp32) {
        cudaDataType cuda_data_type = CUDA_R_32F;
        cusolverDnXsyevdx_workspace_template<float>(
            handle, jobz, range, uplo, N, cuda_data_type, lda, il, iu, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
    } else {
        cudaDataType cuda_data_type = CUDA_R_64F;
        cusolverDnXsyevdx_workspace_template<double>(
            handle, jobz, range, uplo, N, cuda_data_type, lda, il, iu, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
    }

    *workspaceBytesDevice.data_ptr<size_t>() = workspaceInBytesOnDevice;
    *workspaceBytesHost.data_ptr<size_t>() = workspaceInBytesOnHost;
}

void cusolverDnXsyev_batched_export(
    torch::Tensor a, 
    torch::Tensor w,
    torch::Tensor info,
    bool upper_triangle,
    bool eigenvalues_only,
    uintptr_t stream_ptr,
    bool verbose
) {
    signal(SIGINT, signal_handler);
    
    if (a.dtype() == torch::kFloat32) {
        return cusolverDnXsyev_batched_template<float>(
            a, w, info, upper_triangle, eigenvalues_only, stream_ptr, CUDA_R_32F, verbose
        );
    } 

    if (a.dtype() == torch::kFloat64) {
        return cusolverDnXsyev_batched_template<double>(
            a, w, info, upper_triangle, eigenvalues_only, stream_ptr, CUDA_R_64F, verbose
        );
    }

    // If it gets here the dtype isn't supported
    throw std::runtime_error("Tensor needs to have dtype either float32 or float64");
}

void cusolverDnXsyev_batched_workspace_query_export(
    int N,
    int batch_size,
    bool is_fp32,
    torch::Tensor workspaceBytesDevice,
    torch::Tensor workspaceBytesHost
) {
    signal(SIGINT, signal_handler);

    if (workspaceBytesDevice.dtype() != torch::kUInt64 || workspaceBytesHost.dtype() != torch::kUInt64) {
        throw std::runtime_error("Workspace sizes needs to have dtype uint64");
    }

    if (workspaceBytesDevice.numel() != 1 || workspaceBytesHost.numel() != 1) {
        throw std::runtime_error("Workspace sizes need to have only one element");
    }

    // These don't appear to affect the workspace sizes
    bool upper_triangle = true;
    bool eigenvalues_only = false;

    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();
    // because pytorch is row major and cublas is column major, the triangle is flipped
    cublasFillMode_t uplo = upper_triangle ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cusolverEigMode_t jobz = eigenvalues_only ? CUSOLVER_EIG_MODE_NOVECTOR : CUSOLVER_EIG_MODE_VECTOR;

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    int lda = N;

    if (is_fp32) {
        cudaDataType cuda_data_type = CUDA_R_32F;
        cusolverDnXsyev_batched_workspace_template<float>(
            handle, jobz, uplo, N, batch_size, cuda_data_type, lda, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
    } else {
        cudaDataType cuda_data_type = CUDA_R_64F;
        cusolverDnXsyev_batched_workspace_template<double>(
            handle, jobz, uplo, N, batch_size, cuda_data_type, lda, &workspaceInBytesOnDevice, &workspaceInBytesOnHost
        );
    }

    *workspaceBytesDevice.data_ptr<size_t>() = workspaceInBytesOnDevice;
    *workspaceBytesHost.data_ptr<size_t>() = workspaceInBytesOnHost;
}