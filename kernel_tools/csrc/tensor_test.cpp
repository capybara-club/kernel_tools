#include <torch/extension.h>
#include <stdio.h>

void tensor_test(torch::Tensor a) {
    printf("something: %d\n", a.size(0)+10);
}