#pragma once

#include <torch/extension.h>

void tensor_stats(torch::Tensor a, bool *is_transpose, int *leading_dimension);