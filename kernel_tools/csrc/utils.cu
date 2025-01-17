#include "kernel_tools.h"

// This function looks at sizes and strides of the tensor to determine if the tensor
// is transposed or normal. Pytorch is row major and cublas/fortran is column major.
// The stride of a tensor is how many elements you need to walk to find the next 
// element in the dimension of the stride. So for pytorch being row major, you typically
// need to walk 1 element to find the next column, and the number of columns to find the next 
// row. If the matrix is transposed, the data remains the same, but now the number of elements
// to walk to find the next row is 1 and the number of elements to walk to find the next column
// is the number of columns of the original non-transposed tensor. However, if the tensor is a view
// of a bigger tensor, then the stride would reflect how many elements you need to walk to find the next 
// element of the dimension knowing that the tensor lives inside of the larger tensor. This leads to cases
// where the stride in a dimension can be much larger than the dimension of a tensor.
// In the cublas api, this stride is called the leading dimension of the tensor and is named something like
// 'lda', 'ldc'...
void tensor_stats(torch::Tensor a, bool *is_transpose, int *leading_dimension) {
    int num_dims = a.dim();
    if (num_dims != 2 && num_dims != 3) {
        throw std::runtime_error("Dimension of tensor needs to be 2 or 3");
    }

    if (num_dims == 2) {
        int d_y = a.size(0);
        int d_x = a.size(1);

        int s_y = a.stride(0);
        int s_x = a.stride(1);

        if (s_y != 1 && s_x != 1) {
            throw std::runtime_error("Neither of the tensor strides are 1, this is a strange tensor layout");
        }

        if (d_y == 1 && s_y == 1 && s_x == 1) {
            *is_transpose = false;
            *leading_dimension = 1;
            return;
        }

        if (s_x == 1) {
            *is_transpose = true;
            *leading_dimension = s_y;
            return;
        }

        *is_transpose = false;
        *leading_dimension = s_x;
    } else {
        int batch = a.size(0);
        int d_y = a.size(1);
        int d_x = a.size(2);

        int s_batch = a.stride(0);
        int s_y = a.stride(1);
        int s_x = a.stride(2);

        if (s_y != 1 && s_x != 1) {
            throw std::runtime_error("Neither of the tensor strides are 1, this is a strange tensor layout");
        }

        if (d_y == 1 && s_y == 1 && s_x == 1) {
            *is_transpose = false;
            *leading_dimension = 1;
            return;
        }

        if (s_x == 1) {
            *is_transpose = true;
            *leading_dimension = s_y;
            return;
        }

        *is_transpose = false;
        *leading_dimension = s_x;
    }

    // printf("%d, %d, %d, %d\n", d_y, d_x, s_y, s_x);
}