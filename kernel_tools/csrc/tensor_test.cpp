#include <torch/extension.h>
#include <stdio.h>

void tensor_test(torch::Tensor a) {
    printf("something: %ld\n", a.size(0)+10);
}