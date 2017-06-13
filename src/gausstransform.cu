#include <cub/cub.cuh>



#ifndef block_size_x
    #define block_size_x 128    //best for GTX 690
#endif

/*
 * This function performs the main body of work for computing the Gauss transform
 * The parallelization is such that one thread block is created
 * for each item in A, which is of size m. This implies that each thread block
 * does n (size of B) work.
 * The gradient computed in this function is reduced to a single value within the
 * thread block. The same is done for the cross term, which then needs to be
 * reduced in a second kernel. 
 */
template<typename T, int dim>
__device__ void GaussTransform_blocked_i(const T *A, const T *B,
                 int m, int n, T scale_sq, T *d_grad, T *d_cross_term) {

    int tx = threadIdx.x;

    // Specialize BlockReduce for a 1D block of block_size_x threads on type T
    typedef cub::BlockReduce<T, block_size_x> BlockReduce;
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
    for (int j = tx; j<n; j+=block_size_x) {

        T dist_ij = 0;
        for (int d = 0; d < dim; ++d) {
            dist_ij += (A[i * dim + d] - B[j * dim + d])*(A[i * dim + d] - B[j * dim + d]);
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

    if (tx == 0 && blockIdx.x < m) {
        for (int d = 0; d < dim; d++) {
            d_grad[blockIdx.x * dim + d] = grad_i[d] / (scale_sq * m * n);
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

/*
 * Reduce the per thread block cross terms computed in the GaussTransform kernel to single value
 * and divide by (m*n)
 *
 * This kernel is designed to run as single-thread block, because the number of terms to reduce is
 * of size n or m, which is expected to be around 2000 or so. The number of items to reduce
 * is passed as the last argument 'nblocks', which corresponds to the number of thread blocks used
 * by the first kernel.
 */
extern "C"
__global__ void reduce_cross_term(double *output, double *d_cross_term, int m, int n, int nblocks) {

    int tx = threadIdx.x;
    // Specialize BlockReduce for a 1D block of block_size_x threads on type T
    typedef cub::BlockReduce<double, block_size_x> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double cross_term = 0.0;
    for (int i=tx; i<nblocks; i+=block_size_x) {
        cross_term += d_cross_term[i];
    }

    //reduce to single value within thread block
    cross_term = BlockReduce(temp_storage).Sum(cross_term);

    //thread 0 writes output
    if (tx == 0) {
        output[0] = cross_term / (m*n);
    }

}



/*
 * Host part for calling GPUGaussTransform from the CPU 
 */

class GPUGaussTransform {
    public:
        GPUGaussTransform(int max_n);
        ~GPUGaussTransform();
        double compute(const double *A, const double *B, int m, int n, double scale, double *grad);
    private:
        int dim;
        int max_n;
        double *d_A;
        double *d_B;
        double *d_grad;
        double *d_cross_term;

        cudaStream_t stream;
};


GPUGaussTransform::GPUGaussTransform(int n) {
    //allocate GPU memory for size max_n
    max_n = n;
    dim = 2;
    int elems = max_n * dim;

    cudaError_t err;

    err = cudaMalloc((void **)&d_A, elems*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void **)&d_B, elems*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void **)&d_grad, elems*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void **)&d_cross_term, max_n*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();
} 

GPUGaussTransform::~GPUGaussTransform() {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_grad);
    cudaFree(d_cross_term);
    cudaStreamDestroy(stream);
} 

double GPUGaussTransform::compute(const double *A, const double *B,
    int m, int n, double scale, double *grad) {

    double energy;
    cudaError_t err;

    //move data to the GPU
    err = cudaMemcpyAsync(d_A, A, m*dim*sizeof(double), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyAsync: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpyAsync(d_B, B, n*dim*sizeof(double), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyAsync: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    //setup kernel execution parameters
    dim3 threads(block_size_x, 1, 1);
    dim3 grid(m, 1, 1);
    
    //call the first kernel
    double scale_sq = scale * scale;
    GaussTransform<<<grid, threads, 0, stream>>>(d_A, d_B, m, n, scale_sq, d_grad, d_cross_term); 

    //call the second kernel
    dim3 grid2(1, 1, 1);
    reduce_cross_term<<<grid2, threads, 0, stream>>>(d_cross_term, d_cross_term, m, n, m);

    //copy result from GPU memory to host memory
    err = cudaMemcpyAsync(grad, d_grad, m*dim*sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyDeviceToHost: %s\n", cudaGetErrorString (err));
        exit(1);
    }

    err = cudaMemcpyAsync(&energy, d_cross_term, 1*sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyDeviceToHost: %s\n", cudaGetErrorString (err));
        exit(1);
    }

    return energy;
}



extern "C"
float test_GaussTransformHost(double *cost, const double* A, const double* B,
            int m, int n, int dim, double scale, double* grad) {

    GPUGaussTransform gpu_gt(m);

    *cost = gpu_gt.compute(A, B, m, n, scale, grad);

    return 0.0;
}


