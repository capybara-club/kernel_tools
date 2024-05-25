/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <torch/extension.h>
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverMg.h>

#include "cusolverMg_utils.h"
#include "cusolver_utils.h"

#include "kernel_tools.h"

template <typename T>
void cusolverMgSyevd_workspace_template(
    int N,
    int num_devices,
    bool use_num_devices_visible,
    int64_t *workspace_elements,
    bool verbose
) {
    using data_type = T;

    const int MAX_NUM_DEVICES = 16;

    std::vector<int> deviceList(MAX_NUM_DEVICES);

    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; /* tile size */
    const int lda = N;

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    cudaLibMgMatrixDesc_t descrA = NULL;
    cudaLibMgGrid_t gridA = NULL;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    int64_t lwork = 0; /* workspace: number of elements per device */

    int nbGpus = num_devices;
    if (use_num_devices_visible) {
        CUDA_CHECK( cudaGetDeviceCount(&nbGpus) );

        nbGpus = (nbGpus < MAX_NUM_DEVICES) ? nbGpus : MAX_NUM_DEVICES;
        if (verbose) std::printf("\tThere are %d GPUs \n", nbGpus);
        for (int j = 0; j < nbGpus; j++) {
            deviceList[j] = j;
            cudaDeviceProp prop;
            CUDA_CHECK( cudaGetDeviceProperties(&prop, j) );
            if (verbose) std::printf("\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor);
        }
    }

    cusolverMgHandle_t cusolver_mg_handle = NULL;
    CUSOLVER_CHECK( cusolverMgCreate(&cusolver_mg_handle) );
    CUSOLVER_CHECK( cusolverMgDeviceSelect(cusolver_mg_handle, nbGpus, deviceList.data()) );
    CUSOLVER_CHECK( cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping) );

    /* (global) A is N-by-N */
    CUSOLVER_CHECK(
        cusolverMgCreateMatrixDesc(
            &descrA, 
            N, /* nubmer of rows of (global) A */
            N,          /* number of columns of (global) A */
            N,          /* number or rows in a tile */
            T_A,        /* number of columns in a tile */
            traits<data_type>::cuda_data_type, 
            gridA
        )
    );

    CUSOLVER_CHECK( 
        cusolverMgSyevd_bufferSize(
            cusolver_mg_handle, 
            jobz, 
            CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
            N, 
            NULL,
            IA,         /* base-1 */
            JA,         /* base-1 */
            descrA, 
            NULL,
            traits<data_type>::cuda_data_type,
            traits<data_type>::cuda_data_type, 
            &lwork
        )
    );

    if (verbose) std::printf("\tAllocate device workspace, lwork = %lld \n", static_cast<long long>(lwork));
    *workspace_elements = lwork;

    if (descrA != NULL) CUSOLVER_CHECK( cusolverMgDestroyMatrixDesc(descrA) );
    if (gridA != NULL) CUSOLVER_CHECK( cusolverMgDestroyGrid(gridA) );

    CUSOLVER_CHECK( cusolverMgDestroy(cusolver_mg_handle) );
}

template <typename T>
void cusolverMgSyevd_template(
    torch::Tensor a, 
    torch::Tensor d,
    int max_devices,
    bool verbose
) {
    using data_type = T;

    /* maximum number of GPUs */
    const int MAX_NUM_DEVICES = max_devices;

    int nbGpus = 0;
    std::vector<int> deviceList(MAX_NUM_DEVICES);

    const int N = a.size(0);
    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; /* tile size */
    const int lda = N;

    int info = 0;

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    cudaLibMgMatrixDesc_t descrA = NULL;
    cudaLibMgGrid_t gridA = NULL;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    int64_t lwork = 0; /* workspace: number of elements per device */

    CUDA_CHECK( cudaSetDevice(0) );
    cusolverMgHandle_t cusolver_mg_handle = NULL;
    CUSOLVER_CHECK( cusolverMgCreate(&cusolver_mg_handle) );

    if (verbose) std::printf("Step 1: Create Mg handle and select devices \n");
    CUDA_CHECK( cudaGetDeviceCount(&nbGpus) );
    CUDA_CHECK( cudaDeviceSynchronize() );

    nbGpus = (nbGpus < MAX_NUM_DEVICES) ? nbGpus : MAX_NUM_DEVICES;
    if (verbose) std::printf("\tThere are %d GPUs \n", nbGpus);
    for (int j = 0; j < nbGpus; j++) {
        deviceList[j] = j;
        cudaDeviceProp prop;
        CUDA_CHECK( cudaGetDeviceProperties(&prop, j) );
        if (verbose) std::printf("\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor);
    } 

    CUSOLVER_CHECK( cusolverMgDeviceSelect(cusolver_mg_handle, nbGpus, deviceList.data()) );

    if (verbose) std::printf("step 2: Enable peer access.\n");
    CUDA_CHECK( enablePeerAccess(nbGpus, deviceList.data()) );

    if (verbose) std::printf("Step 5: Create matrix descriptors for A and D \n");
    CUSOLVER_CHECK( cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping) );

    /* (global) A is N-by-N */
    CUSOLVER_CHECK(
        cusolverMgCreateMatrixDesc(
            &descrA, 
            N, /* nubmer of rows of (global) A */
            N,          /* number of columns of (global) A */
            N,          /* number or rows in a tile */
            T_A,        /* number of columns in a tile */
            traits<data_type>::cuda_data_type, gridA
        )
    );

    if (verbose) std::printf("Step 8: Allocate workspace space \n");
    CUSOLVER_CHECK( 
        cusolverMgSyevd_bufferSize(
            cusolver_mg_handle, 
            jobz, 
            CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
            N, 
            NULL, //reinterpret_cast<void **>(array_d_A.data()), 
            IA,         /* base-1 */
            JA,         /* base-1 */
            descrA, 
            NULL, // reinterpret_cast<void *>(d_data), 
            traits<data_type>::cuda_data_type,
            traits<data_type>::cuda_data_type, 
            &lwork
        )
    );

    if (verbose) std::printf("\tAllocate device workspace, lwork = %lld \n", static_cast<long long>(lwork));
    if (verbose) std::printf("Step 6: Allocate distributed matrices A and D \n");
    std::vector<data_type *> array_d_A(nbGpus, nullptr);
    data_type *a_data = a.data_ptr<data_type>();
    data_type *d_data = d.data_ptr<data_type>();

    std::vector<data_type *> array_d_work(nbGpus, nullptr);

    cudaError_t cuda_status = cudaSuccess;
    cusolverStatus_t cuda_solver_status = CUSOLVER_STATUS_SUCCESS;

    // Hacky way to make sure on a bad status we still deallocate all the memory
    do {
        /* A := 0 */
        cuda_status = 
            createMat<data_type>(
                nbGpus, 
                deviceList.data(), 
                N, /* number of columns of global A */
                T_A,                          /* number of columns per column tile */
                lda,                          /* leading dimension of local A */
                array_d_A.data()
            );
        if (cuda_status != cudaSuccess) break;

        if (verbose) std::printf("Step 7: Prepare data on devices \n");
        cuda_status = 
            memcpyH2D<data_type>(nbGpus, deviceList.data(), N, N,
                /* input */
                a_data, lda,
                /* output */
                N,                /* number of columns of global A */
                T_A,              /* number of columns per column tile */
                lda,              /* leading dimension of local A */
                array_d_A.data(), /* host pointer array of dimension nbGpus */
                IA, JA
            );
        if (cuda_status != cudaSuccess) break;

        /* array_d_work[j] points to device workspace of device j */
        cuda_status = 
            workspaceAlloc(
                nbGpus, 
                deviceList.data(),
                sizeof(data_type) * lwork, /* number of bytes per device */
                reinterpret_cast<void **>(array_d_work.data())
            );
        if (cuda_status != cudaSuccess) break;

        /* sync all devices */
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) break;

        if (verbose) std::printf("Step 9: Compute eigenvalues and eigenvectors \n");
        cuda_solver_status = 
            cusolverMgSyevd(
                cusolver_mg_handle, 
                jobz, 
                CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
                N, 
                reinterpret_cast<void **>(array_d_A.data()),             /* exit: eigenvectors */
                IA, 
                JA, 
                descrA, 
                reinterpret_cast<void **>(d_data),        /* exit: eigenvalues */
                traits<data_type>::cuda_data_type, 
                traits<data_type>::cuda_data_type,
                reinterpret_cast<void **>(array_d_work.data()), 
                lwork, 
                &info /* host */
            );
        if (cuda_solver_status != CUSOLVER_STATUS_SUCCESS) break;

        /* sync all devices */
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) break;

        /* check if SYEVD converges */
        if (0 > info) {
            // Break here and check info after everything is freed
            break;
        }

        if (verbose) std::printf("Step 10: Copy eigenvectors to A and eigenvalues to D\n");
        cuda_status = 
            memcpyD2H<data_type>(nbGpus, deviceList.data(), N, N,
                /* input */
                N,   /* number of columns of global A */
                T_A, /* number of columns per column tile */
                lda, /* leading dimension of local A */
                array_d_A.data(), IA, JA,
                /* output */
                a_data, /* N-y-N eigenvectors */
                lda
            );
        if (cuda_status != cudaSuccess) break;
    } while(0);

    if (verbose) std::printf("step 12: Free resources \n");
    CUDA_CHECK( 
        destroyMat(
            nbGpus, 
            deviceList.data(), 
            N,   /* number of columns of global A */
            T_A, /* number of columns per column tile */
            reinterpret_cast<void **>(array_d_A.data())
        )
    );

    CUDA_CHECK( 
        workspaceFree(
            nbGpus, 
            deviceList.data(), 
            reinterpret_cast<void **>(array_d_work.data())
        ) 
    );

    if (descrA != NULL) CUSOLVER_CHECK( cusolverMgDestroyMatrixDesc(descrA) );
    if (gridA != NULL) CUSOLVER_CHECK( cusolverMgDestroyGrid(gridA) );
    CUSOLVER_CHECK( cusolverMgDestroy(cusolver_mg_handle) );

    if (0 > info) {
        char buffer[100];
        std::snprintf(buffer, sizeof(buffer), "%d-th parameter is wrong \n", -info);
        throw std::runtime_error(buffer);
    }

    CUDA_CHECK(cuda_status);
    CUSOLVER_CHECK(cuda_solver_status);
}

static void signal_handler(int signum) {
    exit(signum);
}

void cusolverMgSyevd_export(
    torch::Tensor a, 
    torch::Tensor d,
    int max_devices,
    bool verbose
) {
    signal(SIGINT, signal_handler);

    if (a.dtype() != d.dtype())
        throw std::runtime_error("Both tensors must have same dtype");

    if (a.is_cuda() || d.is_cuda())
        throw std::runtime_error("Tensors must be on the Host, not device.");

    if (a.dim() != 2) 
        throw std::runtime_error("Dimension of tensor needs to be 2");

    if (a.size(0) != a.size(1)) 
        throw std::runtime_error("Matrix needs to be square");

    // Need to do this because there is a bug in cusolverMg handle creation/destruction. Pending fix...
    // This shouldn't leak memory even with bug. I ran this 2000 times in a row watching vram usage.
    // pid_t pid = fork();
    // if (pid == 0) { // This is the child process
    if (a.dtype() != torch::kFloat32 && a.dtype() != torch::kFloat64) {
        throw std::runtime_error("Tensor needs to have dtype either float32 or float64");
    }

    if (a.dtype() == torch::kFloat32) {
        cusolverMgSyevd_template<float>(a, d, max_devices, verbose);
    } else {
        cusolverMgSyevd_template<double>(a, d, max_devices, verbose);
    }
    //     _exit(0); // Exit child process
    // } else {
    //     int status;
    //     waitpid(pid, &status, 0);
    // }
}

void cusolverMgSyevd_workspace_query_export(
    int N,
    int num_devices,
    bool is_fp32,
    bool use_num_devices_visible,
    torch::Tensor workspace_num_elements,
    bool verbose
) {
    signal(SIGINT, signal_handler);

    if (workspace_num_elements.dtype() != torch::kInt64) {
        throw std::runtime_error("workspace_num_elements tensor needs to have dtype int64");
    }

    if (workspace_num_elements.numel() != 1) {
        throw std::runtime_error("workspace_num_elements tensor needs to have only one element");
    }

    int64_t num_workspace_elements;

    // Need to do this because there is a bug in cusolverMg handle creation/destruction. Pending fix...
    // pid_t pid = fork();
    // if (pid == 0) { // This is the child process
    if (is_fp32) {
        cusolverMgSyevd_workspace_template<float>(N, num_devices, use_num_devices_visible, &num_workspace_elements, verbose);
    } else {
        cusolverMgSyevd_workspace_template<double>(N, num_devices, use_num_devices_visible, &num_workspace_elements, verbose);
    }
    *workspace_num_elements.data_ptr<int64_t>() = num_workspace_elements;
    //     _exit(0); // Exit child process
    // } else {
    //     int status;
    //     waitpid(pid, &status, 0);
    // }
}