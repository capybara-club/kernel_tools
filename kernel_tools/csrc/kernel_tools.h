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

void cusolverMgSyevd_export(
    torch::Tensor a, 
    torch::Tensor d,
    bool verbose,
    bool dry_run
);

void cusolverMgSyevd_workspace_query_export(
    int N,
    int num_devices,
    bool is_fp32,
    bool verbose
);
