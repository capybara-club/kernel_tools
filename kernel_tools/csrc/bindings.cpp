#include <torch/extension.h>

#include <tensor_test.h>

// void tensor_test(torch::Tensor a) {
//     printf("tensor: %d\n", a.size(0)+10);
// }

// Bindings for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_test", &tensor_test, "tensor_test");
}