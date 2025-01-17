#include <torch/extension.h>

#include "kernel_tools.h"

// Bindings for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cusolverDnXsyevdx_export", &cusolverDnXsyevdx_export, "cusolverDnXsyevdx_export");
    m.def("cusolverDnXsyevdx_workspace_query_export", &cusolverDnXsyevdx_workspace_query_export, "cusolverDnXsyevdx_workspace_query_export");
    m.def("cusolverDnXsyev_batched_export", &cusolverDnXsyev_batched_export, "cusolverDnXsyev_batched_export");
    m.def("cusolverDnXsyev_batched_workspace_query_export", &cusolverDnXsyev_batched_workspace_query_export, "cusolverDnXsyev_batched_workspace_query_export");
    m.def("cusolverMgSyevd_export", &cusolverMgSyevd_export, "cusolverMgSyevd_export");
    m.def("cusolverMgSyevd_workspace_query_export", &cusolverMgSyevd_workspace_query_export, "cusolverMgSyevd_workspace_query_export");
}