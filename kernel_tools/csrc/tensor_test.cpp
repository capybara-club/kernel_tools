#include <torch/extension.h>
#include <stdio.h>

void tensor_test(torch::Tensor a) {
    printf("tensor: %ld\n", a.size(0));
}