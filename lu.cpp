#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cblas.h>
#include <lapacke.h>

#include "util.h"

#include "MatrixView.hpp"

using namespace mblas;

int dgetrf(int m, int n, double* a, int lda, int* ipiv)
{
    constexpr int nb = 128;

    MatrixView A(a, m, n, lda);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    for (int j = 0; j < n; j += nb)
    {
        int jb = std::min(nb, n-j);

        auto panel = A.subview(j, j, m-j, jb);

        LAPACKE_dgetf2(LAPACK_COL_MAJOR, panel.nrows, panel.ncols, panel.data, panel.stride, &ipiv[j]);

        auto leftA = A.subview(j, 0, m-j, j);
        auto rightA = A.subview(j, j+jb, m-j, n-j-jb);

        dlaswp(cublas, leftA, 1, jb, &ipiv[j], 1);
        dlaswp(cublas, rightA, 1, jb, &ipiv[j], 1);

        for (int i = 0; i < jb; i++)
            ipiv[j + i] += j;
        
        if (n - (j+jb) > 0) {
            auto tileA = A.subview(j, j, jb, jb);
            auto U = A.subview(j, j+jb, jb, n - (j+jb));
            auto L = A.subview(j+jb, j, m - (j+jb), jb);
            auto restA = A.subview(j+jb, j+jb, m - (j+jb), n - (j+jb));

            dtrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, 
                  1.0, tileA, U);
            
            dgemm(cublas, -1.0, L, U, 1.0, restA);

            cudaDeviceSynchronize();
        }
    }
    return 0;
}


int main() {
    int m = 8000;
    int n = m;
    int lda = m;

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
    if (m < 5000)
    {
        Acopy = (double*)malloc(data_size);
        CUDA_CALL(cudaMemcpy(Acopy, A, data_size, cudaMemcpyDefault));
    }

    // Run LU factorization
    //
    int* ipiv = (int*)malloc(sizeof(int) * m);
    

    double startTime = get_real_time();
    dgetrf(m, n, A, m, ipiv);
    double endTime = get_real_time();
    
    printf("matrix_size=%d,time_s=%f\n", n, endTime-startTime);

    // Extract the solution vector
    //
    if (m < 5000)
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

