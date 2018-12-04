#pragma once
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include <time.h>


#define CUDA_CALL(call) { cudaAssert((call), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDAError:%s:%d: %s\n", file, line, cudaGetErrorString(code));
        abort();
    }
}

#define CURAND_CALL(call) { curandAssert((call), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char* file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRAND Error:%s:%d\n", file, line);
        abort();
    }
}

#define CUBLAS_CALL(call) { cublasAssert((call), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char* file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas Error:%s:%d\n", file, line);
        abort();
    }
}

double get_real_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec/1000000000.0;
}

/// Implementation of the dlaswp LAPACK function in style of cublas.
///
cublasStatus_t cublasDlaswp(cublasHandle_t handle, 
                            int m, int n,
                            double* a, int lda, 
                            int k1, int k2, 
                            int* ipiv, int incx) {

    if (n == 0) return CUBLAS_STATUS_SUCCESS;

    for (int i = k1-1; i < k2; i++) {
        cublasStatus_t err = cublasDswap(handle, n, &a[i], lda, &a[ipiv[i]-1], lda);
        if (err != CUBLAS_STATUS_SUCCESS)
            return err;
    }
    return CUBLAS_STATUS_SUCCESS;
}

