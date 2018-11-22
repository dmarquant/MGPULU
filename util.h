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

