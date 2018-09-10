#pragma once

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

double get_real_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec/1000000000.0;
}

