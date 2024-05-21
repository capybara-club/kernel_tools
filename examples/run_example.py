import torch

from kernel_tools import warmup, tensor_test

warmup()
a = torch.arange(10)
tensor_test(a)