#include <cub/cub.cuh>



#define SQR(X)  ((X)*(X))

template<typename T>
T GaussTransform(const T* A, const T* B,
    int m, int n, int dim, T scale) {
  T cross_term = 0;
  scale = SQR(scale);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      T dist_ij = 0;
      for (int d = 0; d < dim; ++d) {
        dist_ij += SQR(A[i * dim + d] - B[j * dim + d]);
      }
      T cost_ij = exp(-1.0 * dist_ij / scale);
      cross_term += cost_ij;
    }
    /* printf("cross_term = %.3f\n", cross_term);  */
  }
  return cross_term / (m * n);
}





#ifndef block_size_x
    #define block_size_x 256
#endif





template<typename T, int dim>
__device__ void GaussTransform_blocked_i(const T *A, const T *B,
                 int m, int n, T scale_sq, T *d_grad, T *d_cross_term) {

    // Specialize BlockReduce for a 1D block of block_size_x threads on type float
    typedef cub::BlockReduce<float, block_size_x> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T cross_term = 0.0;
    T grad_i[dim];
    for (int d = 0; d < dim; d++) {
        grad_i[d] = 0.0;
    }

    //for (int i = 0; i < m; ++i) { //loop parallelized over threads blocks
    int i = blockIdx.x;
    if (i>=m) return;

    //loop parallelized over threads within thread block
    for (int j = threadIdx.x; j<n; j+=block_size_x) {

        T dist_ij = 0;
        for (int d = 0; d < dim; ++d) {
            dist_ij += SQR(A[i * dim + d] - B[j * dim + d]);
        }
        T cost_ij = exp(-1.0 * dist_ij / scale_sq);

        for (int d = 0; d < dim; ++d) {
            grad_i[d] -= cost_ij * 2.0 * (A[i * dim + d] - B[j * dim + d]);
        }

        cross_term += cost_ij;
    }

    //reduce grad_i for each d, within the block (division by scale^2*m*n on CPU)
    for (int d = 0; d < dim; d++) {
        grad_i[d] = BlockReduce(temp_storage).Sum(grad_i[d]);
    }

    //reduce cross_term within the block, (division by m*n on CPU)
    cross_term = BlockReduce(temp_storage).Sum(cross_term);


    if (tx == 0) {
        for (int d = 0; d < dim; d++) {
            d_grad[blockIdx.x * dim + d] = grad_i[d];
        }
        d_cross_term[blockIdx.x] = cross_term;
    }
}


extern "C"
__global__ void GaussTransform(const double* A, const double* B,
                 int m, int n, double scale_sq, double *grad, double *cross_term) {

    //2-dimensional with double precision
    GaussTransform_blocked_i<double, 2>(A, B, m, n, scale_sq, grad, cross_term);

}

