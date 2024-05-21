#include <torch/extension.h>

#include "kernel_tools.h"

// Bindings for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_test", &tensor_test, "tensor_test");
    m.def("cusolverDnXsyevdx_export", &cusolverDnXsyevdx_export, "cusolverDnXsyevdx_export");
    m.def("cusolverMgSyevd_export", &cusolverMgSyevd_export, "cusolverMgSyevd_export");
}