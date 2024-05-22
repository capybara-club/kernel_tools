#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>

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

    printf("cusolverDnXsyevdx workspace requested: device: %zu bytes, host: %zu bytes\n", *workspaceInBytesOnDevice, *workspaceInBytesOnHost);
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
    cudaDataType_t cuda_data_type
) {
    cusolverDnHandle_t handle = get_cusolver_dn_handle().get();
    // because pytorch is row major and cublas is column major, the triangle is flipped
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

    printf("cusolverDnXsyevdx workspace requested: device: %zu bytes, host: %zu bytes\n", workspaceInBytesOnDevice, workspaceInBytesOnHost);

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

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Failed to run cusolverDnXsyevdx: " << status << std::endl;
        throw std::runtime_error("Failed to run cusolverDnXsyevdx");
    }

    cuda_free(cuda_data);
    host_free(host_data);
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
    uintptr_t stream_ptr
) {
    if (a.dtype() == torch::kFloat32) {
        return cusolverDnXsyevdx_template<float>(
            a, w, info, il, iu, upper_triangle, eigenvalues_only, eigen_range, stream_ptr, CUDA_R_32F
        );
    } 

    if (a.dtype() == torch::kFloat64) {
        return cusolverDnXsyevdx_template<double>(
            a, w, info, il, iu, upper_triangle, eigenvalues_only, eigen_range, stream_ptr, CUDA_R_64F
        );
    }

    // If it gets here the dtype isn't supported
    throw std::runtime_error("Tensor needs to have dtype either float32 or float64");
}

