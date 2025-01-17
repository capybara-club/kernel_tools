#pragma once

#include <torch/extension.h>

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
);

void cusolverDnXsyevdx_workspace_query_export(
    int N,
    bool is_fp32,
    torch::Tensor workspaceBytesDevice,
    torch::Tensor workspaceBytesHost
);

void cusolverDnXsyev_batched_export(
    torch::Tensor a, 
    torch::Tensor w,
    torch::Tensor info,
    bool upper_triangle,
    bool eigenvalues_only,
    uintptr_t stream_ptr,
    bool verbose
);

void cusolverDnXsyev_batched_workspace_query_export(
    int N,
    int batch_size,
    bool is_fp32,
    torch::Tensor workspaceBytesDevice,
    torch::Tensor workspaceBytesHost
);

void cusolverMgSyevd_export(
    torch::Tensor a, 
    torch::Tensor d,
    int max_devices,
    bool verbose
);

void cusolverMgSyevd_workspace_query_export(
    int N,
    int num_devices,
    bool is_fp32,
    bool use_num_devices_visible,
    torch::Tensor workspaceNumElements,
    bool verbose
);
