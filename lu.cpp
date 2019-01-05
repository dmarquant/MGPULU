#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cblas.h>
#include <lapacke.h>

#include "Range.hpp"
#include "util.h"

void sync_all(int ngpus)
{
    for (int gpu = 0; gpu < ngpus; gpu++) {
        CUDA_CALL(cudaSetDevice(gpu));
        CUDA_CALL(cudaDeviceSynchronize());
    }
}

/// Copy a rectangular area from a 2d array in newly allocated memory.
///
double* copy_rect(double* a, int lda, int nrows, int ncols, cudaStream_t stream) {
    if (nrows == 0 || ncols == 0) return nullptr;

    double* b;
    CUDA_CALL(cudaMalloc(&b, sizeof(double) * nrows * ncols));

    int ldb = nrows * sizeof(double);
    CUDA_CALL(cudaMemcpy2DAsync(b, ldb, a, lda * sizeof(double),
                                sizeof(double) * nrows, ncols,
                                cudaMemcpyDefault, stream));
    return b;
}

/// Copy a 2d array into an existing region of another array.
/// This is the counter part of copy_rect.
///
void copy_rect_back(double* a, int lda, double* b, int ldb, int nrows, int ncols, cudaStream_t stream) {
    if (nrows == 0 || ncols == 0) return;

    CUDA_CALL(cudaMemcpy2DAsync(a, lda * sizeof(double), b, ldb * sizeof(double),
                                sizeof(double) * nrows, ncols,
                                cudaMemcpyDefault, stream));
}


/// Multi GPU implementation of LU decomposition. The input matrix is split horizontally
/// among the GPUs. During each iteration the slices are copied local to the GPUs.
///
int loopfusion_dgetrf(int ngpus, cudaStream_t* streams, cublasHandle_t* handles, int m, int n, double* a, int lda, int* ipiv)
{
    constexpr int nb = 1024;

    for (int j = 0; j < n; j += nb)
    {

        int jb = std::min(nb, n-j);
        LAPACKE_dgetrf(LAPACK_COL_MAJOR, m-j, jb, &a[j + j*lda], lda, &ipiv[j]);

        // Divide the parts of A left and right to the current panel in equal parts
        // 
        std::vector<Range> left_range = partition(0, j, ngpus);
        std::vector<Range> right_range = partition(j+jb, n, ngpus);

        // Allocate local buffers 
        // 
        double* panel[ngpus];
        double* left_aslices[ngpus];
        double* right_aslices[ngpus];

        for (int gpu = 0; gpu < ngpus; gpu++) {
            Range left  = left_range[gpu];
            Range right = right_range[gpu];

            CUDA_CALL(cudaSetDevice(gpu));
            panel[gpu]         = copy_rect(&a[j + j*lda],           lda, m-j, jb, streams[gpu]);
            left_aslices[gpu]  = copy_rect(&a[j + left.begin*lda],  lda, m-j, left.size(), streams[gpu]);
            right_aslices[gpu] = copy_rect(&a[j + right.begin*lda], lda, m-j, right.size(), streams[gpu]);
        }

        // The leading dimension of the local arrays is the same for all local arrays.
        int ld = (m-j);

        double ONE = 1.0;
        double MINUS_ONE = -1.0;
        for (int gpu = 0; gpu < ngpus; gpu++) {
            Range left  = left_range[gpu];
            Range right = right_range[gpu];

            CUDA_CALL(cudaSetDevice(gpu));
            CUBLAS_CALL(cublasDlaswp(handles[gpu], m-j, left.size(),
                                     left_aslices[gpu], ld,
                                     1, jb, &ipiv[j], 1));

            CUBLAS_CALL(cublasDlaswp(handles[gpu], m-j, right.size(),
                                     right_aslices[gpu], ld,
                                     1, jb, &ipiv[j], 1));

            if (right.size()) {
                CUBLAS_CALL(cublasDtrsm(handles[gpu], CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                            jb, right.size(),
                            &ONE,
                            panel[gpu], ld,
                            right_aslices[gpu], ld));

                CUBLAS_CALL(cublasDgemm(handles[gpu], CUBLAS_OP_N, CUBLAS_OP_N,
                            m-(j+jb), right.size(), jb,
                            &MINUS_ONE,
                            &panel[gpu][jb], ld,
                            right_aslices[gpu], ld,
                            &ONE,
                            &right_aslices[gpu][jb], ld));
            }
        }

        // Copy back data
        //
        for (int gpu = 0; gpu < ngpus; gpu++) {
            Range left  = left_range[gpu];
            Range right = right_range[gpu];

            CUDA_CALL(cudaSetDevice(gpu));
            copy_rect_back(&a[j + left.begin*lda],  lda, left_aslices[gpu], ld, m-j, left.size(), streams[gpu]);
            copy_rect_back(&a[j + right.begin*lda], lda, right_aslices[gpu], ld, m-j, right.size(), streams[gpu]);

            CUDA_CALL(cudaFree(left_aslices[gpu]));
            CUDA_CALL(cudaFree(right_aslices[gpu]));
        }

        sync_all(ngpus);

        // Update the pivoted rows to global row indices
        for (int i = 0; i < jb; i++)
            ipiv[j + i] += j;

    }
    return 0;
}


int main(int argc, char** argv) {
    constexpr int TEST_SIZE = 10000;

    int m = 8192;
    if (argc >= 2)
        m = std::stoi(argv[1]);

    int n = m;
    int lda = m;

    int ngpus;
    CUDA_CALL(cudaGetDeviceCount(&ngpus));

    cudaStream_t streams[ngpus];
    cublasHandle_t handles[ngpus];

    for (int gpu = 0; gpu < ngpus; gpu++) {
        CUDA_CALL(cudaSetDevice(gpu));
        CUDA_CALL(cudaStreamCreate(&streams[gpu]));
        CUBLAS_CALL(cublasCreate(&handles[gpu]));
        CUBLAS_CALL(cublasSetStream(handles[gpu], streams[gpu]));
    }
    CUDA_CALL(cudaSetDevice(0));


    double* A;
    double* Acopy;
    double* b;

    size_t data_size = sizeof(double) * lda * n;
    CUDA_CALL(cudaMallocManaged(&A, data_size));
    CUDA_CALL(cudaMallocManaged(&b, sizeof(double) * m));


    // Generate random coefficient matrix and result vector
    //
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 2343ULL));
    CURAND_CALL(curandGenerateUniformDouble(gen, A, lda*n));
    CURAND_CALL(curandGenerateUniformDouble(gen, b, m));
    CURAND_CALL(curandDestroyGenerator(gen));

    // Copy A for later reference
    //
    if (m < TEST_SIZE)
    {
        Acopy = (double*)malloc(data_size);
        CUDA_CALL(cudaMemcpy(Acopy, A, data_size, cudaMemcpyDefault));
    }

    // Run LU factorization
    //
    int* ipiv = (int*)malloc(sizeof(int) * m);
    

    double startTime = get_real_time();
    loopfusion_dgetrf(ngpus, streams, handles, m, n, A, m, ipiv);
    double endTime = get_real_time();
    
    printf("%s,%d,%d,%f\n", argv[0], ngpus, n, endTime-startTime);

    // Extract the solution vector
    //
    if (m < TEST_SIZE)
    {
        double* x = (double*)malloc(sizeof(double) * n);
        CUDA_CALL(cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDefault));

        double* Ahost = (double*)malloc(data_size);
        CUDA_CALL(cudaMemcpy(Ahost, A, data_size, cudaMemcpyDefault));
        LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', n, 1, Ahost, lda, ipiv, x, n);

        // Check the solution
        //
        double* b2 = (double*)malloc(sizeof(double) * n);
        cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1.0, Acopy, lda,
                    x, 1, 0.0, b2, 1);

        for (int i = 0; i < n; i++) {
            double x = b[i];
            double y = b2[i];
            if (!(std::abs(x-y) < 0.00001)) {
                printf("Result vectors are not equal:\n");
                printf("b[%d] = %f | b2[%d] = %f\n", i, b[i], i, b2[i]);
                return 0;
            }
        }
    }
    return 0;
}

