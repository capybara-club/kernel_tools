import torch
from kernel_tools._C import tensor_test

def test():
    a = torch.arange(10)
    tensor_test(a)
    print('dogs')

print('cats')
test()