#include <math.h>

//declaration
//template<typename T>
//T GaussTransform(const T* A, const T* B, int m, int n,
//    int dim, T scale, T* grad);

//definition
#include "gausstransform_ref.cpp"

//compile-time instantation
//double GaussTransform(const double* A, const double* B, int m, int n, int dim, double 
//        scale, double* grad);

//extern C interface
extern "C" {

float call_GaussTransform(double* cost, const double* A, const double* B, int m, int n, int dim, double 
        scale, double* grad) {
    *cost = GaussTransform(A, B, m, n, dim, scale, grad);
}


}

