#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cblas.h>
#include <lapacke.h>

#include "util.h"


/// Single GPU implementation of LU decomposition.
///
int dgetrf(cublasHandle_t cublas, int m, int n, double* a, int lda, int* ipiv) {
    // Block size
    constexpr int nb = 512;

    for (int j = 0; j < n; j += nb) {
        // The last block might be smaller
        int jb = std::min(nb, n-j);

        // Do a partial LU decomposition on the current panel
        LAPACKE_dgetrf(LAPACK_COL_MAJOR, m-j, jb, &a[j + j*lda], lda, &ipiv[j]);

        // Update the pivoted rows to global row indices
        for (int i = 0; i < jb; i++)
            ipiv[j + i] += j;

        // Apply row swaps to left and right of the matrix
        CUBLAS_CALL(cublasDlaswp(cublas, m-j, j,         a,             lda, j+1, j+jb, ipiv, 1));
        CUBLAS_CALL(cublasDlaswp(cublas, m-j, n-(j+jb), &a[(j+jb)*lda], lda, j+1, j+jb, ipiv, 1));

        if (n - (j+jb) > 0) {
            double ONE = 1.0;
            double MINUS_ONE = -1.0;

            // Update U
            CUBLAS_CALL(cublasDtrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                                    CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                                    jb, n-(j+jb),
                                    &ONE,
                                    &a[j + j*lda], lda,
                                    &a[j + (j+jb)*lda], lda));

            // Update rest of A
            CUBLAS_CALL(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m-(j+jb), n-(j+jb), jb,
                                    &MINUS_ONE,
                                    &a[j+jb + j*lda], lda,
                                    &a[j + (j+jb)*lda], lda,
                                    &ONE,
                                    &a[j+jb + (j+jb)*lda], lda));

            CUDA_CALL(cudaDeviceSynchronize());
        }
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

    cudaStream_t stream;
    cublasHandle_t handle;

    CUDA_CALL(cudaStreamCreate(&stream));
    CUBLAS_CALL(cublasCreate(&handle));
    CUBLAS_CALL(cublasSetStream(handle, stream));

    double* A;
    double* Acopy = nullptr;
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
    dgetrf(handle, m, n, A, m, ipiv);
    double endTime = get_real_time();
    
    printf("ngpus=%d,matrix_size=%d,time_s=%f\n", 1, n, endTime-startTime);

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

