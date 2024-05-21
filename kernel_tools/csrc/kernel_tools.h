#pragma once

#include <torch/extension.h>

void cusolverDnXsyevdx_export(
    torch::Tensor a, 
    torch::Tensor w,
    torch::Tensor info,
    int num_eigs,
    bool upper_triangle,
    uintptr_t stream_ptr
);

void cusolverMgSyevd_export(
    torch::Tensor a, 
    torch::Tensor d
);
