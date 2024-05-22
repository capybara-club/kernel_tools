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
    uintptr_t stream_ptr
);

void cusolverMgSyevd_export(
    torch::Tensor a, 
    torch::Tensor d
);
